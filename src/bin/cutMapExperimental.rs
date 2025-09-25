// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! This is an experimental interactive cutting-then-mapping
//! implementation that only builds original partitions without processing them.
//!
//! The key idea is to only repartition the endpoint groups that
//! are unable to be mapped, but instead of processing partitions,
//! it just dumps the original partitions.

use std::path::PathBuf;
use gem::repcut::RCHyperGraph;
use gem::aigpdk::AIGPDKLeafPins;
use gem::aig::AIG;
use gem::staging::build_staged_aigs;
use gem::pe::{process_partitions_conditional_skip, Partition};
use netlistdb::NetlistDB;
use rayon::prelude::*;

// Global constant for maximum number of partitions before considering convergence
const TOO_MANY_PARTITIONS: usize = 2000;

/// Call builtin partitioner.
fn run_par(hg: &RCHyperGraph, num_parts: usize) -> Vec<Vec<usize>> {
    clilog::debug!("invoking partitioner (#parts {})", num_parts);
    let parts_ids = hg.partition(num_parts);
    let mut parts = vec![vec![]; num_parts];
    for (i, part_id) in parts_ids.into_iter().enumerate() {
        parts[part_id].push(i);
    }
    parts
}

#[derive(clap::Parser, Debug)]
struct SimulatorArgs {
    /// Gate-level verilog path synthesized in our provided library.
    ///
    /// If your design is still at RTL level, you should synthesize it
    /// in yosys first.
    netlist_verilog: PathBuf,
    /// Top module type in netlist to analyze.
    ///
    /// If not specified, we will guess it from the hierarchy.
    #[clap(long)]
    top_module: Option<String>,
    /// Level split thresholds.
    #[clap(long, value_delimiter=',')]
    level_split: Vec<usize>,
    /// Output path for the serialized partitions.
    parts_out: PathBuf,
    /// The maximum allowance of layers for merging-induced degradations.
    ///
    /// By default is 0, meaning no degradation is allowed.
    #[clap(long, default_value_t=0)]
    max_stage_degrad: usize,
}

fn main() {
    clilog::init_stderr_color_debug();
    clilog::set_max_print_count(clilog::Level::Warn, "NL_SV_LIT", 1);
    clilog::info!("Rayon detected {} parallel threads", rayon::current_num_threads());
    let args = <SimulatorArgs as clap::Parser>::parse();
    clilog::info!("Simulator args:\n{:#?}", args);

    let netlistdb = NetlistDB::from_sverilog_file(
        &args.netlist_verilog,
        args.top_module.as_deref(),
        &AIGPDKLeafPins()
    ).expect("cannot build netlist");

    let aig = AIG::from_netlistdb(&netlistdb);
    println!("netlist has {} pins, {} aig pins, {} and gates",
             netlistdb.num_pins, aig.num_aigpins, aig.and_gate_cache.len());

    let stageds = build_staged_aigs(&aig, &args.level_split);

    let stages_effective_parts = stageds.iter().map(|&(l, r, ref staged)| {
        clilog::info!("interactive partitioning stage {}-{}", l, match r {
            usize::MAX => "max".to_string(),
            r @ _ => format!("{}", r)
        });

        // always made sure that staged output pins are at fronts.
        // Helper function to extract maximum partition depth
        fn get_max_partition_depth(partitions: &[Partition]) -> usize {
            partitions.iter()
                .map(|part| part.stages.len())
                .max()
                .unwrap_or(0)
        }

        // Helper function to perform one iteration of partitioning
        fn perform_partitioning_iteration(
            aig: &AIG,
            staged: &gem::staging::StagedAIG,
            estimated_nodes_per_part: usize,
        ) -> (Vec<Vec<usize>>, Vec<Partition>) {
            let mut unrealized_endpoints = (0..staged.num_endpoint_groups()).collect::<Vec<_>>();
            let mut parts_indices_good: Vec<Vec<usize>> = Vec::new();
            
            // Calculate total number of nodes in the current staged element
            let mut all_endpoint_pins = Vec::new();
            for &endpoint_id in &unrealized_endpoints {
                let endpoint_group = staged.get_endpoint_group(&aig, endpoint_id);
                endpoint_group.for_each_input(|pin| {
                    all_endpoint_pins.push(pin);
                });
            }
            let total_order = aig.topo_traverse_generic(
                Some(&all_endpoint_pins),
                staged.primary_inputs.as_ref()
            );
            let total_nodes = total_order.len();
            
            // Calculate target_parts based on total nodes (ceiling division)
            let target_parts = (total_nodes + estimated_nodes_per_part - 1) / estimated_nodes_per_part;
            
            let mut division = if target_parts > 0 {
                unrealized_endpoints.len() / target_parts
            } else {
                unrealized_endpoints.len()
            };
            clilog::info!("One iteration's division calculated as: {} (endpoints: {}, total_nodes: {}, target_parts: {}, estimated_nodes_per_part: {})", 
                         division, unrealized_endpoints.len(), total_nodes, target_parts, estimated_nodes_per_part);

            while !unrealized_endpoints.is_empty() {
                division = (division / 2).max(1);
                let num_parts = (unrealized_endpoints.len() + division - 1) / division;
                clilog::info!("current: {} endpoints, try {} parts", unrealized_endpoints.len(), num_parts);
                let staged_ur = staged.to_endpoint_subset(&unrealized_endpoints);
                let hg_ur = RCHyperGraph::from_staged_aig(&aig, &staged_ur);
                let mut parts_indices = run_par(&hg_ur, num_parts);
                for idcs in &mut parts_indices {
                    for i in idcs {
                        *i = unrealized_endpoints[*i];
                    }
                }
                let parts_try = parts_indices.par_iter()
                    .map(|endpts| Partition::build_one(&aig, staged, endpts))
                    .collect::<Vec<_>>();
                let mut new_unrealized_endpoints = Vec::new();
                for (idx, part_opt) in parts_indices.into_iter().zip(parts_try.into_iter()) {
                    match part_opt {
                        Some(_part) => {
                            parts_indices_good.push(idx);
                        }
                        None => {
                            if idx.len() == 1 {
                                panic!("A single endpoint still cannot map, you need to increase level cut granularity.");
                            }
                            for endpt_i in idx {
                                new_unrealized_endpoints.push(endpt_i);
                            }
                        }
                    }
                }
                new_unrealized_endpoints.sort_unstable();
                unrealized_endpoints = new_unrealized_endpoints;
            }

            clilog::info!("interactive partition completed for one iteration: {} in total. building original partitions.",
                          parts_indices_good.len());

            // Build original partitions
            let all_original_parts = {
                clilog::info!("Building original partitions in parallel");
                parts_indices_good.par_iter().enumerate().map(|(i, v)| {
                    let part = Partition::build_one(&aig, staged, v);
                    if part.is_none() {
                        clilog::error!("Partition {} exceeds resource constraint.", i);
                    }
                    part
                }).collect::<Vec<_>>()
            };

            // Filter out None values and unwrap the Some values
            let original_parts: Vec<Partition> = all_original_parts.into_iter()
                .filter_map(|part_opt| part_opt)
                .collect();

            clilog::info!("built {} original partitions.", original_parts.len());
            (parts_indices_good, original_parts)
        }

        // Adaptive partitioning loop
        let mut estimated_nodes_per_part = 40000;
        let mut previous_max_depth = usize::MAX;
        let mut best_parts_indices = Vec::new();
        let mut previous_estimated_nodes_per_part = estimated_nodes_per_part;
        let mut previous_parts_indices = Vec::new();
        let mut converged_due_to_too_many = false;
        
        loop {
            clilog::info!("Starting partitioning iteration with ESTIMATED_NODES_PER_PART = {}", estimated_nodes_per_part);
            
            let (parts_indices, original_parts) = perform_partitioning_iteration(&aig, staged, estimated_nodes_per_part);
            let current_max_depth = get_max_partition_depth(&original_parts);
            let current_num_partitions = parts_indices.len();
            
            clilog::info!("Current iteration maxPartDepth: {}, num_partitions: {}", current_max_depth, current_num_partitions);
            
            // Check convergence: stop if depth doesn't improve
            if previous_max_depth != usize::MAX && current_max_depth >= previous_max_depth {
                clilog::info!("Convergence reached. Previous maxPartDepth: {}, Current maxPartDepth: {}", 
                             previous_max_depth, current_max_depth);
                break;
            }
            
            // Check new convergence criteria: too many partitions and depth not significantly better
            if current_num_partitions > TOO_MANY_PARTITIONS && 
               previous_max_depth != usize::MAX && 
               current_max_depth >= previous_max_depth.saturating_sub(1) {
                clilog::info!("Convergence reached due to too many partitions ({} > {}) and depth not significantly better (current: {}, previous: {})", 
                             current_num_partitions, TOO_MANY_PARTITIONS, current_max_depth, previous_max_depth);
                clilog::info!("Re-running partitioning with previous iteration settings (maxPartDepth: {}, estimated_nodes_per_part: {})", 
                             previous_max_depth, previous_estimated_nodes_per_part);
                // Re-run partitioning with previous settings to get fresh partition indices
                estimated_nodes_per_part = previous_estimated_nodes_per_part;
                let (fresh_parts_indices, fresh_original_parts) = perform_partitioning_iteration(&aig, staged, estimated_nodes_per_part);
                let fresh_max_depth = get_max_partition_depth(&fresh_original_parts);
                clilog::info!("Fresh partitioning result: maxPartDepth: {}, num_partitions: {}", fresh_max_depth, fresh_parts_indices.len());
                best_parts_indices = fresh_parts_indices;
                converged_due_to_too_many = true;
                break;
            }
            
            // Update best results
            previous_parts_indices = best_parts_indices.clone();
            previous_estimated_nodes_per_part = estimated_nodes_per_part;
            best_parts_indices = parts_indices;
            previous_max_depth = current_max_depth;
            
            // Reduce estimated_nodes_per_part by half for next iteration
            estimated_nodes_per_part /= 2;
            
            // Safety check to prevent infinite loop
            if estimated_nodes_per_part < 1000 {
                clilog::warn!("ESTIMATED_NODES_PER_PART became too small ({}), stopping adaptive loop", estimated_nodes_per_part);
                break;
            }
        }
        
        // Final iteration with the previous (better) value
        // Only perform final iteration if we haven't already converged due to too many partitions
        if estimated_nodes_per_part < 40000 && !best_parts_indices.is_empty() && !converged_due_to_too_many {
            estimated_nodes_per_part *= 2; // Restore to previous value
            clilog::info!("Performing final partitioning iteration with ESTIMATED_NODES_PER_PART = {}", estimated_nodes_per_part);
            let (final_parts_indices, final_original_parts) = perform_partitioning_iteration(&aig, staged, estimated_nodes_per_part);
            let final_max_depth = get_max_partition_depth(&final_original_parts);
            
            clilog::info!("Final iteration maxPartDepth: {}", final_max_depth);
            
            // Use the better result
            if final_max_depth < previous_max_depth {
                clilog::info!("Using final iteration results (better depth: {} vs {})", final_max_depth, previous_max_depth);
                best_parts_indices = final_parts_indices;
            } else {
                clilog::info!("Using previous iteration results (better depth: {} vs {})", previous_max_depth, final_max_depth);
            }
        } else if best_parts_indices.is_empty() {
            clilog::warn!("No valid partitions found, using empty partition list");
        }
        
        clilog::info!("Adaptive partitioning completed. Processing partitions with max_stage_degrad = {}", args.max_stage_degrad);
        
        let effective_parts = process_partitions_conditional_skip(
            &aig, staged, best_parts_indices, args.max_stage_degrad
        ).unwrap();
        clilog::info!("after merging: {} parts.", effective_parts.len());
        effective_parts
    }).collect::<Vec<_>>();

    let f = std::fs::File::create(&args.parts_out).unwrap();
    let mut buf = std::io::BufWriter::new(f);
    serde_bare::to_writer(&mut buf, &stages_effective_parts).unwrap();
}
