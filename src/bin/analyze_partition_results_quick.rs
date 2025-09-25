// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Quick analysis script for partition results from cut_map_interactive.rs
//!
//! This script reads the output file from cut_map_interactive.rs and provides
//! quick analysis of the resulting partitions without detailed endpoint group analysis.
//!
//! Usage:
//!   cargo run --bin analyze_partition_results_quick -- --gem-parts-file parts.bin
//!
//! The script analyzes each partition and reports:
//! 1. Number of BoomerangStages
//! 2. Width and depth of each BoomerangStage
//! 3. Number of endpoints and active nodes
//! 4. BoomerangUtilization percentage

use std::path::PathBuf;
use gem::pe::Partition;
use serde_bare;

#[derive(clap::Parser, Debug)]
struct AnalyzeArgs {
    /// Input file containing partition results from cut_map_interactive.rs
    #[arg(short, long)]
    gem_parts_file: PathBuf,
}

fn main() {
    let args = <AnalyzeArgs as clap::Parser>::parse();
    
    // Read the partition results
    let f = std::fs::File::open(&args.gem_parts_file).expect("Failed to open gem parts file");
    let stages_effective_parts: Vec<Vec<Partition>> = serde_bare::from_reader(f)
        .expect("Failed to deserialize partition results");
    
    println!("=== Quick Partition Analysis Results ===");
    
    for (stage_idx, partitions) in stages_effective_parts.iter().enumerate() {
        println!("Stage {}: {} partitions", stage_idx, partitions.len());
        println!("{}", "=".repeat(50));
        
        for (part_idx, partition) in partitions.iter().enumerate() {
            analyze_partition_quick(part_idx, partition);
            println!();
        }
    }
}

fn analyze_partition_quick(
    part_idx: usize, 
    partition: &Partition
) {
    println!("Partition {}:", part_idx);
    
    // 1. Count BoomerangStages
    let num_boomerang_stages = partition.stages.len();
    println!("  BoomerangStages: {}", num_boomerang_stages);
    
    // 2. Analyze each BoomerangStage
    for (stage_idx, boomerang_stage) in partition.stages.iter().enumerate() {
        analyze_boomerang_stage(stage_idx, boomerang_stage);
    }
    
    // 3. Count endpoints and active nodes
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
    
    // 4. Calculate BoomerangUtilization
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
