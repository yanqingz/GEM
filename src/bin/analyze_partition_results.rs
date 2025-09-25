// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Analysis script for partition results from cut_map_interactive.rs
//!
//! This script reads the output file from cut_map_interactive.rs and provides
//! detailed analysis of the resulting partitions.
//!
//! Usage:
//!   cargo run --bin analyze_partition_results -- --gem-parts-file parts.bin --netlist-gv netlist.gv --level-split 100,200,300
//!   cargo run --bin analyze_partition_results -- --gem-parts-file parts.bin --netlist-gv netlist.gv
//!
//! The script analyzes each partition and reports:
//! 1. Count of each EndpointGroup type (PrimaryOutput, DFF, RAMBlock, StagedIOPin)
//! 2. RAMBlock names (if netlist.gv is provided)
//! 3. Number of BoomerangStages
//! 4. Width and depth of each BoomerangStage
//! 5. Number of endpoints and active nodes
//! 6. BoomerangUtilization percentage

use std::path::PathBuf;
use gem::aig::{AIG, EndpointGroup};
use gem::aigpdk::AIGPDKLeafPins;
use gem::pe::Partition;
use gem::staging::{StagedAIG, build_staged_aigs};
use netlistdb::{NetlistDB, GeneralHierName};
use serde_bare;

#[derive(clap::Parser, Debug)]
struct AnalyzeArgs {
    /// Input file containing partition results from cut_map_interactive.rs
    #[arg(short, long)]
    gem_parts_file: PathBuf,
    
    /// Optional: Original netlist.gv file for RAMBlock name lookup
    #[arg(short, long)]
    netlist_gv: Option<PathBuf>,
    
    /// Level split thresholds (comma-separated). If not provided, treats the entire AIG as one stage.
    #[clap(long, value_delimiter=',')]
    level_split: Vec<usize>,
    
    /// Print DFF names for the deepest stage (specify stage number, 1-indexed)
    #[clap(long)]
    print_deepest_dffs: Option<usize>,
}

fn main() {
    let args = <AnalyzeArgs as clap::Parser>::parse();
    
    // No validation needed - empty level_split is handled by build_staged_aigs
    
    // Read the partition results
    let f = std::fs::File::open(&args.gem_parts_file).expect("Failed to open gem parts file");
    let stages_effective_parts: Vec<Vec<Partition>> = serde_bare::from_reader(f)
        .expect("Failed to deserialize partition results");
    
    // Load the original netlist for AIG reconstruction and RAMBlock names
    let netlistdb = if let Some(netlist_path) = &args.netlist_gv {
        Some(NetlistDB::from_sverilog_file(netlist_path, None, &AIGPDKLeafPins()).expect("Failed to load netlist"))
    } else {
        println!("Warning: No netlist.gv provided. RAMBlock names will not be available.");
        None
    };
    
    // Reconstruct AIG and staged AIGs if netlist is available
    let (aig, stageds) = if let Some(ref netlistdb) = netlistdb {
        let aig = AIG::from_netlistdb(netlistdb);
        // Use the user-provided level split points
        let stageds = build_staged_aigs(&aig, &args.level_split);
        (Some(aig), Some(stageds))
    } else {
        (None, None)
    };
    
    println!("=== Partition Analysis Results ===");
    
    for (stage_idx, partitions) in stages_effective_parts.iter().enumerate() {
        println!("Stage {}: {} partitions", stage_idx, partitions.len());
        println!("{}", "=".repeat(50));
        
        let staged = stageds.as_ref().and_then(|s| s.get(stage_idx).map(|(_, _, staged)| staged));
        
        for (part_idx, partition) in partitions.iter().enumerate() {
            analyze_partition(part_idx, partition, aig.as_ref(), staged, netlistdb.as_ref(), args.print_deepest_dffs);
            println!();
        }
    }
}

fn analyze_partition(
    part_idx: usize, 
    partition: &Partition, 
    aig: Option<&AIG>, 
    staged: Option<&StagedAIG>, 
    netlistdb: Option<&NetlistDB>,
    print_deepest_dffs: Option<usize>
) {
    println!("Partition {}:", part_idx);
    
    // 1. Count EndpointGroups
    let mut primary_outputs = 0;
    let mut dffs = 0;
    let mut ram_blocks = 0;
    let mut staged_io_pins = 0;
    let mut ram_block_names = Vec::new();
    
    if let (Some(aig), Some(staged)) = (aig, staged) {
        for &endpoint_id in &partition.endpoints {
            let endpoint_group = staged.get_endpoint_group(aig, endpoint_id);
            match endpoint_group {
                EndpointGroup::PrimaryOutput(_) => {
                    primary_outputs += 1;
                },
                EndpointGroup::DFF(_) => {
                    dffs += 1;
                },
                EndpointGroup::RAMBlock(_ram_block) => {
                    ram_blocks += 1;
                    // Get RAMBlock name if netlist is available
                    if let Some(netlistdb) = netlistdb {
                        // Find the cell ID for this RAMBlock
                        // This is a simplified approach - you might need to adjust based on your data structure
                        for (cell_id, cell_type) in netlistdb.celltypes.iter().enumerate() {
                            if cell_type.as_str() == "$__RAMGEM_SYNC_" {
                                // This is a RAMBlock cell - get its name
                                let cell_name = &netlistdb.cellnames[cell_id];
                                ram_block_names.push(cell_name.dbg_fmt_hier());
                                break;
                            }
                        }
                    }
                },
                EndpointGroup::StagedIOPin(_) => {
                    staged_io_pins += 1;
                },
            }
        }
    } else {
        println!("  Warning: Cannot analyze EndpointGroups without AIG and StagedAIG");
    }
    
    println!("  EndpointGroups:");
    println!("    PrimaryOutputs: {}", primary_outputs);
    println!("    DFFs: {}", dffs);
    println!("    RAMBlocks: {}", ram_blocks);
    println!("    StagedIOPins: {}", staged_io_pins);
    
    if !ram_block_names.is_empty() {
        println!("  RAMBlock Names:");
        for name in &ram_block_names {
            println!("    {}", name);
        }
    }
    
    // 2. Count BoomerangStages
    let num_boomerang_stages = partition.stages.len();
    println!("  BoomerangStages: {}", num_boomerang_stages);
    
    // 2.5. Print DFF names for specified deepest stage if requested
    if let Some(target_stage) = print_deepest_dffs {
        if num_boomerang_stages >= target_stage {
            println!("  DFF Names ({}th BoomerangStage writeouts):", target_stage);
            let mut dff_names = Vec::new();
            
            if let (Some(aig), Some(staged), Some(netlistdb)) = (aig, staged, netlistdb) {
                // Get the target stage (convert from 1-indexed to 0-indexed)
                if let Some(target_stage_data) = partition.stages.get(target_stage - 1) {
                    // Check if this DFF's writeout occurs in the target stage
                    for &endpoint_id in &partition.endpoints {
                        let endpoint_group = staged.get_endpoint_group(aig, endpoint_id);
                        match endpoint_group {
                            EndpointGroup::DFF(dff) => {
                                // Check if this DFF's d_iv appears in the target stage's writeouts
                                let mut found_in_target_stage = false;
                                
                                // Look through the hierarchy of the target stage to see if dff.d_iv appears
                                for level in &target_stage_data.hier {
                                    for &node in level {
                                        if node != usize::MAX && node == dff.d_iv >> 1 {
                                            found_in_target_stage = true;
                                            break;
                                        }
                                    }
                                    if found_in_target_stage {
                                        break;
                                    }
                                }
                                
                                if found_in_target_stage {
                                    // Find the cell ID for this DFF
                                    for (cell_id, dff_ref) in &aig.dffs {
                                        if dff_ref.d_iv == dff.d_iv && dff_ref.en_iv == dff.en_iv && dff_ref.q == dff.q {
                                            let cell_name = &netlistdb.cellnames[*cell_id];
                                            dff_names.push(cell_name.dbg_fmt_hier());
                                            break;
                                        }
                                    }
                                }
                            },
                            _ => {} // We only care about DFFs
                        }
                    }
                }
            }
            
            if dff_names.is_empty() {
                println!("    No DFFs found with writeouts in the {}th BoomerangStage", target_stage);
            } else {
                for name in &dff_names {
                    println!("    {}", name);
                }
                println!("    Total DFFs in {}th stage: {}", target_stage, dff_names.len());
            }
        } else {
            println!("  Warning: Requested stage {} exceeds available stages ({})", target_stage, num_boomerang_stages);
        }
    }
    
    // 3. Analyze each BoomerangStage
    for (stage_idx, boomerang_stage) in partition.stages.iter().enumerate() {
        analyze_boomerang_stage(stage_idx, boomerang_stage);
    }
    
    // 4. Count endpoints and active nodes
    let num_endpoints = partition.endpoints.len();
    let total_active_nodes = partition.stages.iter()
        .map(|stage| {
            stage.hier.iter()
                .map(|level| level.iter().filter(|&&node| node != usize::MAX).count())
                .sum::<usize>()
        })
        .sum::<usize>();
    
    println!("  Endpoints: {}", num_endpoints);
    println!("  Active Nodes: {}", total_active_nodes);
    
    // 5. Calculate BoomerangUtilization
    let total_boomerang_slots = partition.stages.iter()
        .map(|stage| stage.hier.iter().map(|level| level.len()).sum::<usize>())
        .sum::<usize>();
    
    let boomerang_utilization = if total_boomerang_slots > 0 {
        (total_active_nodes as f64) / (total_boomerang_slots as f64) * 100.0
    } else {
        0.0
    };
    
    println!("  BoomerangUtilization: {:.2}%", boomerang_utilization);
}

fn analyze_boomerang_stage(stage_idx: usize, boomerang_stage: &gem::pe::BoomerangStage) {
    println!("  BoomerangStage {}:", stage_idx);
    
    // Calculate width (max active elements in any level)
    let mut max_width = 0;
    let mut max_depth = 0;
    
    for (level_idx, level) in boomerang_stage.hier.iter().enumerate() {
        let active_count = level.iter().filter(|&&node| node != usize::MAX).count();
        max_width = max_width.max(active_count);
        
        // Check if this level has any active nodes
        if active_count > 0 {
            max_depth = level_idx;
        }
    }
    
    println!("    Width: {}", max_width);
    println!("    Depth: {}", max_depth);
    
    // Additional details about write_outs
    let num_write_outs = boomerang_stage.write_outs.len();
    println!("    Write-outs: {}", num_write_outs);
}
