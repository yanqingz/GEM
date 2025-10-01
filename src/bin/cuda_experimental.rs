// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
use std::path::PathBuf;
use gem::aigpdk::AIGPDKLeafPins;
use gem::aig::{DriverType, AIG};
use gem::staging::build_staged_aigs;
use gem::pe::Partition;
use gem::flatten::FlattenedScriptV1;
use netlistdb::{Direction, GeneralPinName, NetlistDB};
use sverilogparse::SVerilogRange;
use compact_str::CompactString;
use ulib::{AsUPtrMut, Device, UVec};
use std::fs::File;
use std::io::{BufReader, BufWriter, Seek, SeekFrom};
use std::hash::Hash;
use std::rc::Rc;
use std::collections::{HashMap, HashSet};
use vcd_ng::{Parser, ScopeItem, Var, Scope, FastFlow, FastFlowToken, FFValueChange, Writer, SimulationCommand};


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
    /// Input path for the serialized partitions.
    gemparts: PathBuf,
    /// VCD input signal path
    input_vcd: String,
    /// The scope path of top module in the input VCD.
    ///
    /// If not specified, we will use a flat view.
    /// (this view is often incorrect..)
    #[clap(long)]
    input_vcd_scope: Option<String>,
    /// Output VCD path (must be writable)
    output_vcd: String,
    /// The scope path of top module in the output VCD.
    ///
    /// If not specified, we will use `gem_top_module`.
    #[clap(long)]
    output_vcd_scope: Option<String>,
    /// the number of CUDA blocks to map and execute with.
    ///
    /// should not exceed GPU maximum simutaneous occupancy.
    num_blocks: usize,
    /// Whether to run a sanity check against CPU baseline on finish.
    #[clap(long)]
    check_with_cpu: bool,
    /// Limit the number of simulated cycles to no more than this.
    #[clap(long)]
    max_cycles: Option<usize>,
    /// Include all DFF Q outputs in the VCD dump (in addition to primary outputs).
    #[clap(long)]
    vcd_dump_full: bool,
    /// Amount of GPU device memory available in Gigabytes.
    #[clap(long)]
    gpu_memory_gb: f64,
}

/// Hierarchical name representation in VCD.
#[derive(PartialEq, Eq, Clone, Debug)]
struct VCDHier {
    cur: CompactString,
    prev: Option<Rc<VCDHier>>
}

/// Reverse iterator of a [`VCDHier`], yielding cell names
/// from the bottom to the top module.
struct VCDHierRevIter<'i>(Option<&'i VCDHier>);

impl<'i> Iterator for VCDHierRevIter<'i> {
    type Item = &'i CompactString;

    #[inline]
    fn next(&mut self) -> Option<&'i CompactString> {
        let name = self.0?;
        if name.cur.is_empty() {
            return None
        }
        let ret = &name.cur;
        self.0 = name.prev.as_ref().map(|a| a.as_ref());
        Some(ret)
    }
}

impl<'i> IntoIterator for &'i VCDHier {
    type Item = &'i CompactString;
    type IntoIter = VCDHierRevIter<'i>;

    #[inline]
    fn into_iter(self) -> VCDHierRevIter<'i> {
        VCDHierRevIter(Some(self))
    }
}

impl Hash for VCDHier {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        for s in self.iter() {
            s.hash(state);
        }
    }
}

#[allow(dead_code)]
impl VCDHier {
    #[inline]
    fn single(cur: CompactString) -> Self {
        VCDHier { cur, prev: None }
    }

    #[inline]
    fn empty() -> Self {
        VCDHier { cur: "".into(), prev: None }
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.cur.as_str() == "" && self.prev.is_none()
    }

    #[inline]
    fn iter(&self) -> VCDHierRevIter {
        (&self).into_iter()
    }
}

/// Try to match one component in a scope.
/// If succeed, returns the remaining scope (can be None itself indicating
/// all paths matched).
/// If fails, return None.
fn match_scope_path<'i>(mut scope: &'i str, cur: &str) -> Option<&'i str> {
    if scope.len() == 0 { return Some("") }
    if scope.starts_with('/') {
        scope = &scope[1..];
    }
    if scope.len() == 0 { Some("") }
    else if scope.starts_with(cur) {
        if scope.len() == cur.len() { Some("") }
        else if scope.as_bytes()[cur.len()] == b'/' {
            Some(&scope[cur.len() + 1..])
        }
        else { None }
    }
    else { None }
}

fn find_top_scope<'i>(
    items: &'i [ScopeItem], top_scope: &'_ str
) -> Option<&'i Scope> {
    for item in items {
        if let ScopeItem::Scope(scope) = item {
            if let Some(s1) = match_scope_path(
                top_scope, scope.identifier.as_str()
            ) {
                return match s1 {
                    "" => Some(scope),
                    _ => find_top_scope(&scope.children[..], s1)
                };
            }
        }
    }
    None
}


mod ucci {
    include!(concat!(env!("OUT_DIR"), "/uccbind/kernel_v1.rs"));
}


/// Write VCD output for a specific chunk
fn write_vcd_chunk(
    writer: &mut Writer<BufWriter<File>>,
    input_states: &[u32],
    offsets_timestamps: &[(usize, u64)],
    out2vcd: &[(usize, u32, vcd_ng::IdCode)],
    reg_io_state_size: usize,
    chunk_start_cycle: usize
) {
    let mut last_val = vec![2; out2vcd.len()];
    
    for &(global_offset, timestamp) in offsets_timestamps {
        if timestamp == u64::MAX {
            continue
        }
        writer.timestamp(timestamp).unwrap();
        
        // Convert global offset to chunk-relative offset
        // global_offset is the absolute position in the original input_states array
        // We need to subtract the start offset to get the position within the chunk
        let chunk_offset = global_offset - (chunk_start_cycle * reg_io_state_size);
        
        for (i, &(output_aigpin, output_pos, vid)) in out2vcd.iter().enumerate() {
            use vcd_ng::Value;
            let value_new = match output_pos {
                u32::MAX => {
                    assert!(output_aigpin <= 1);
                    output_aigpin as u32
                },
                output_pos @ _ => {
                    // Check if this is an input signal (high bit set)
                    if (output_pos & (1u32 << 31)) != 0 {
                        // This is an input signal, read from input state at current timestamp
                        let input_pos = output_pos & !(1u32 << 31);
                        let value_new_input = input_states[chunk_offset - reg_io_state_size + (input_pos >> 5) as usize] >> (input_pos & 31) & 1;
                        value_new_input
                    } else {
                        // This is an output signal, read from output state
                        let value_new_output = input_states[chunk_offset + (output_pos >> 5) as usize] >> (output_pos & 31) & 1;
                        value_new_output
                    }
                },
            };
            if value_new == last_val[i] {
                continue
            }
            last_val[i] = value_new;
            writer.change_scalar(vid.into(), match value_new {
                1 => Value::V1,
                _ => Value::V0
            }).unwrap();
        }
    }
}

fn main() {
    clilog::init_stderr_color_debug();
    clilog::enable_timer("cuda_experimental");
    clilog::enable_timer("gem");
    clilog::enable_timer("simulation");
    clilog::set_max_print_count(clilog::Level::Warn, "NL_SV_LIT", 1);
    let args = <SimulatorArgs as clap::Parser>::parse();
    clilog::info!("Simulator args:\n{:#?}", args);

    let netlistdb = NetlistDB::from_sverilog_file(
        &args.netlist_verilog,
        args.top_module.as_deref(),
        &AIGPDKLeafPins()
    ).expect("cannot build netlist");

    let aig = AIG::from_netlistdb(&netlistdb);
    let stageds = build_staged_aigs(&aig, &args.level_split);

    let f = std::fs::File::open(&args.gemparts).unwrap();
    let mut buf = std::io::BufReader::new(f);
    let parts_in_stages: Vec<Vec<Partition>> = serde_bare::from_reader(&mut buf).unwrap();
    clilog::info!("# of effective partitions in each stage: {:?}",
                  parts_in_stages.iter().map(|ps| ps.len()).collect::<Vec<_>>());

    let mut input_layout = Vec::new();
    for (i, driv) in aig.drivers.iter().enumerate() {
        if let DriverType::InputPort(_) | DriverType::InputClockFlag(_, _) = driv {
            input_layout.push(i);
        }
    }

    let script = FlattenedScriptV1::from(
        &aig, &stageds.iter().map(|(_, _, staged)| staged).collect::<Vec<_>>(),
        &parts_in_stages.iter().map(|ps| ps.as_slice()).collect::<Vec<_>>(),
        args.num_blocks, input_layout
    );

    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hasher;
    let mut s = DefaultHasher::new();
    script.blocks_data.hash(&mut s);
    println!("Script hash: {}", s.finish());

    // simulate with the script.
    let input_vcd = File::open(&args.input_vcd).unwrap();
    let mut bufrd = BufReader::with_capacity(65536, input_vcd);
    let mut vcd_parser = Parser::new(&mut bufrd);
    let header = vcd_parser.parse_header().unwrap();
    drop(vcd_parser);
    let mut vcd_file = bufrd.into_inner();
    vcd_file.seek(SeekFrom::Start(0)).unwrap();
    let mut vcdflow = FastFlow::new(vcd_file, 65536);

    let top_scope = find_top_scope(
        &header.items[..],
        args.input_vcd_scope.as_deref().unwrap_or("")
    ).expect("Specified top scope not found in VCD.");

    let mut vcd2inp = HashMap::new();
    let mut inp_port_given = HashSet::new();

    let mut match_one_input = |var: &Var, i: Option<isize>, vcd_pos: usize| {
        let key = (VCDHier::empty(), var.reference.as_str(), i);
        if let Some(&id) = netlistdb.pinname2id.get(
            &key as &dyn GeneralPinName
        ) {
            if netlistdb.pindirect[id] != Direction::O { return }
            vcd2inp.insert((var.code.0, vcd_pos), id);
            inp_port_given.insert(id);
        }
    };
    for scope_item in &top_scope.children[..] {
        if let ScopeItem::Var(var) = scope_item {
            use vcd_ng::ReferenceIndex::*;
            match var.index {
                None => match var.size {
                    1 => match_one_input(var, None, 0),
                    w @ _ => {
                        for (pos, i) in (0..w).rev()
                            .enumerate()
                        {
                            match_one_input(
                                var, Some(i as isize), pos)
                        }
                    }
                },
                Some(BitSelect(i)) => match_one_input(
                    var, Some(i as isize), 0),
                Some(Range(a, b)) => {
                    for (pos, i) in SVerilogRange(
                        a as isize, b as isize).enumerate()
                    {
                        match_one_input(var, Some(i), pos);
                    }
                }
            }
        }
    }
    for i in netlistdb.cell2pin.iter_set(0) {
        if netlistdb.pindirect[i] != Direction::I &&
            !inp_port_given.contains(&i)
        {
            clilog::warn!(
                GATESIM_VCDI_MISSING_PI,
                "Primary input port {:?} not present in \
                 the VCD input",
                netlistdb.pinnames[i]);
        }
    }

    // open out
    let write_buf = File::create(&args.output_vcd).unwrap();
    let write_buf = BufWriter::new(write_buf);
    let mut writer = Writer::new(write_buf);
    if let Some((ratio, unit)) = header.timescale {
        writer.timescale(ratio, unit).unwrap();
    }
    let output_vcd_scope = args.output_vcd_scope.as_deref().unwrap_or("gem_top_module");
    let output_vcd_scope = output_vcd_scope.split('/').collect::<Vec<_>>();
    for &scope in &output_vcd_scope {
        writer.add_module(scope).unwrap();
    }
    let mut out2vcd = netlistdb.cell2pin.iter_set(0).filter_map(|i| {
        if netlistdb.pindirect[i] == Direction::I {
            let aigpin = aig.pin2aigpin_iv[i];
            if matches!(aig.drivers[aigpin >> 1], DriverType::InputPort(_)) {
                clilog::info!("skipped output for port {} as it is a pass-through of input port.", netlistdb.pinnames[i].dbg_fmt_pin());
                return None
            }
            if aigpin <= 1 {
                return Some((aigpin, u32::MAX, writer.add_wire(
                    1, &format!("{}", netlistdb.pinnames[i].dbg_fmt_pin())).unwrap()))
            }
            Some((aigpin, *script.output_map.get(&aigpin).unwrap(), writer.add_wire(
                1, &format!("{}", netlistdb.pinnames[i].dbg_fmt_pin())).unwrap()))
        }
        else { None }
    }).collect::<Vec<_>>();

    // Add DFF Q outputs to VCD output if --vcd-dump-full is specified
    // Only add Q pins for DFFs whose instance names are NOT created by Yosys
    let mut dff_count = 0;
    if args.vcd_dump_full {
        use regex::Regex;
        let signal_pattern = Regex::new(r"_[0-9]+__reg").unwrap();
        
        for (cellid, dff) in &aig.dffs {
            // Check if this DFF's instance name contains _[0-9]+__reg pattern
            let cell_name = &netlistdb.cellnames[*cellid].cur;
            if signal_pattern.is_match(cell_name.as_str()) {
                continue; // Skip DFFs with _[0-9]+__reg pattern in their instance name
            }
            
            
            if dff.q != 0 {
                // Find the netlist pin corresponding to this DFF's Q output
                for pinid in netlistdb.cell2pin.iter_set(*cellid) {
                    if netlistdb.pinnames[pinid].1.as_str() == "Q" {
                        // Use dff.q directly instead of pin2aigpin_iv[pinid]
                        // because dff.q is the actual AIG pin used in simulation
                        let aigpin = dff.q;
                        if let Some(&input_pos) = script.input_map.get(&aigpin) {
                            let pin_name = netlistdb.pinnames[pinid].dbg_fmt_pin();
                            
                            // Remove the ":Q" suffix if it exists
                            let mut wire_name = if pin_name.ends_with(":Q") {
                                pin_name[..pin_name.len()-2].to_string()
                            } else {
                                pin_name.to_string()
                            };
                            
                            // Trim whitespace
                            wire_name = wire_name.trim().to_string();
                            
                            // Remove "_reg" but keep array indices
                            // Use regex to capture the array indices after "_reg" and reconstruct without "_reg"
                            use regex::Regex;
                            let reg_pattern = Regex::new(r"_reg(\[[0-9:\[\]]+\]*)$").unwrap();
                            if let Some(captures) = reg_pattern.captures(&wire_name) {
                                // If we found "_reg" followed by array indices, replace with just the array indices
                                let array_indices = &captures[1];
                                wire_name = reg_pattern.replace(&wire_name, array_indices).to_string();
                            } else if wire_name.ends_with("_reg") {
                                // If it just ends with "_reg" (no array indices), remove it
                                wire_name = wire_name.trim_end_matches("_reg").to_string();
                            }
                            
                            let vid = writer.add_wire(1, &wire_name).unwrap();
                            // Q pins are in input_map, so we need to read from input state
                            out2vcd.push((aigpin, input_pos | (1u32 << 31), vid));
                            dff_count += 1;
                        }
                        break;
                    }
                }
            }
        }
        clilog::info!("Added {} DFF Q outputs to VCD (excluding instances with _[0-9]+__reg pattern)", dff_count);
    }


    for _ in 0..output_vcd_scope.len() {
        writer.upscope().unwrap();
    }
    writer.enddefinitions().unwrap();
    writer.begin(SimulationCommand::Dumpvars).unwrap();

    // do simulation - original approach
    let mut state = vec![0; script.reg_io_state_size as usize];

    // the simulator keeps 2 previous timestamps.
    // vcd_time: the last seen timestamp.
    // vcd_time_last_active: the last timestamp strictly before vcd_time that has
    // active events (e.g., watched clock posedge).
    //
    // when a new timestamp arrives and vcd_time has active events, we simulate
    // the circuit with {actived edge flags from vcd_time}, but do NOT include the
    // input port value changes. then, we associate the result output port values to
    // vcd_time_last_active.
    //
    // the above complexity rises from the necessity to emulate {update, then propagate}
    // behavior from our actual {propagate, then update} implementation.
    let mut vcd_time_last_active = u64::MAX;
    let mut vcd_time = 0;
    let mut last_vcd_time_active = true;
    let mut delayed_bit_changes = HashSet::new();

    let mut input_states = Vec::new();
    let mut offsets_timestamps = Vec::new();

    while let Some(tok) = vcdflow.next_token().unwrap() {
        match tok {
            FastFlowToken::Timestamp(t) => {
                if t == vcd_time { continue }
                if last_vcd_time_active {
                    // clilog::debug!("simulating t={}", vcd_time);
                    input_states.extend(state.iter().copied());
                    offsets_timestamps.push((input_states.len(), vcd_time_last_active));
                    // reset for next timestamp
                    for (_, &(pe, ne)) in &aig.clock_pin2aigpins {
                        if pe != usize::MAX {
                            let p = *script.input_map.get(&pe).unwrap();
                            state[p as usize >> 5] &= !(1 << (p & 31));
                        }
                        if ne != usize::MAX {
                            let p = *script.input_map.get(&ne).unwrap();
                            state[p as usize >> 5] &= !(1 << (p & 31));
                        }
                    }
                    if let Some(max_cycles) = args.max_cycles {
                        if offsets_timestamps.len() >= max_cycles {
                            clilog::info!("reached maximum cycles, stop reading input vcd");
                            break
                        }
                    }
                }
                if last_vcd_time_active {
                    vcd_time_last_active = vcd_time;
                }
                vcd_time = t;
                last_vcd_time_active = false;

                for pos in std::mem::take(&mut delayed_bit_changes) {
                    state[(pos >> 5) as usize] ^= 1u32 << (pos & 31);
                }
            },
            FastFlowToken::Value(FFValueChange { id, bits }) => {
                for (pos, b) in bits.iter().enumerate() {
                    if let Some(&pin) = vcd2inp.get(&(id.0, pos)) {
                        let aigpin = aig.pin2aigpin_iv[pin];
                        assert_eq!(aigpin & 1, 0);
                        let aigpin = aigpin >> 1;
                        let pos = match script.input_map.get(&aigpin).copied() {
                            Some(pos) => pos,
                            None => {
                                panic!("input pin {:?} (netlist id {}, aigpin {}) not found in output map.", netlistdb.pinnames[pin].dbg_fmt_pin(), pin, aigpin);
                            }
                        };
                        let old_value = state[(pos >> 5) as usize] >> (pos & 31) & 1;
                        if old_value == match b { b'1' => 1, _ => 0 } {
                            continue
                        }
                        if let Some((pe, ne)) = aig.clock_pin2aigpins.get(&pin).copied() {
                            if pe != usize::MAX && old_value == 0 {
                                last_vcd_time_active = true;
                                let p = *script.input_map.get(&pe).unwrap();
                                state[p as usize >> 5] |= 1 << (p & 31);
                            }
                            if ne != usize::MAX && old_value == 1 {
                                last_vcd_time_active = true;
                                let p = *script.input_map.get(&ne).unwrap();
                                state[p as usize >> 5] |= 1 << (p & 31);
                            }
                        }
                        delayed_bit_changes.insert(pos);
                    }
                }
            }
        }
    }
    input_states.extend(state.iter().copied());
    clilog::info!("total number of cycles: {}", offsets_timestamps.len());
    
    // Calculate GPU memory usage
    let total_cycles = offsets_timestamps.len();
    let input_states_size = total_cycles * script.reg_io_state_size as usize * 4; // 4 bytes per u32
    let sram_storage_size = script.sram_storage_size as usize * 4; // 4 bytes per u32
    let blocks_data_size = script.blocks_data.len() * 4; // 4 bytes per u32
    let blocks_start_size = script.blocks_start.len() * 8; // 8 bytes per usize
    let total_gpu_memory = input_states_size + sram_storage_size + blocks_data_size + blocks_start_size;
    
    clilog::info!("GPU Memory Usage:");
    clilog::info!("  Input states: {} MB ({:?} bytes)", input_states_size / (1024*1024), input_states_size);
    clilog::info!("  SRAM storage: {} MB ({:?} bytes)", sram_storage_size / (1024*1024), sram_storage_size);
    clilog::info!("  Script data: {} MB ({:?} bytes)", blocks_data_size / (1024*1024), blocks_data_size);
    clilog::info!("  Script indices: {} MB ({:?} bytes)", blocks_start_size / (1024*1024), blocks_start_size);
    clilog::info!("  Total GPU memory: {} MB ({:?} bytes)", total_gpu_memory / (1024*1024), total_gpu_memory);
    
    // Calculate number of chunks based on GPU memory (now mandatory)
    // Convert GPU memory to bytes and subtract 2GB buffer
    let total_mem_bytes = ((args.gpu_memory_gb - 2.0) * 1024.0 * 1024.0 * 1024.0) as usize;
    
    // Calculate available memory for input states after subtracting fixed allocations
    let available_memory = total_mem_bytes - sram_storage_size - blocks_data_size - blocks_start_size;
    
    // Calculate number of chunks using the provided formula:
    // ceiling{[(TOTAL_MEM-2) * 1024^3 - sram_storage_size - blocks_data_size - blocks_start_size] / (script.reg_io_state_size * 4) / total_cycles}
    let num_chunks = if total_cycles > 0 {
        let memory_per_cycle = script.reg_io_state_size as usize * 4;
        let cycles_per_chunk = available_memory / memory_per_cycle;
        if cycles_per_chunk > 0 {
            // Ceiling division: (total_cycles + cycles_per_chunk - 1) / cycles_per_chunk
            (total_cycles + cycles_per_chunk - 1) / cycles_per_chunk
        } else {
            total_cycles // If memory is very limited, use 1 cycle per chunk
        }
    } else {
        1
    };
    
    let cycles_per_chunk = (total_cycles + num_chunks - 1) / num_chunks; // Ceiling division
    let chunk_input_states_size = cycles_per_chunk * script.reg_io_state_size as usize * 4;
    let chunk_total_memory = chunk_input_states_size + sram_storage_size + blocks_data_size + blocks_start_size;
    
    clilog::info!("GPU Memory: {:.1} GB (with 2GB buffer: {:.1} GB)", args.gpu_memory_gb, args.gpu_memory_gb - 2.0);
    clilog::info!("Available memory for input states: {} MB ({:?} bytes)", available_memory / (1024*1024), available_memory);
    clilog::info!("Automatic chunking: {} chunks, {} cycles per chunk", num_chunks, cycles_per_chunk);
    clilog::info!("  Memory per chunk: {} MB ({:?} bytes)", chunk_total_memory / (1024*1024), chunk_total_memory);
    
    // Initialize device and prepare for chunked simulation
    let device = Device::CUDA(0);
    let mut sram_storage = UVec::new_zeroed(script.sram_storage_size as usize, device);
    let mut sram_state_cpu = vec![0u32; script.sram_storage_size as usize];
    
    // Initialize state for chunking - starts with all zeros for first chunk
    let mut current_final_state = vec![0u32; script.reg_io_state_size as usize];
    
    // Process simulation in chunks
    for chunk_idx in 0..num_chunks {
        let start_cycle = chunk_idx * cycles_per_chunk;
        let end_cycle = if chunk_idx == num_chunks - 1 {
            total_cycles // Last chunk goes to the end
        } else {
            (chunk_idx + 1) * cycles_per_chunk
        };
        let actual_end_cycle = std::cmp::min(end_cycle, offsets_timestamps.len());
        let chunk_size = actual_end_cycle - start_cycle;
        
        clilog::info!("Processing chunk {}/{}: cycles {} to {} ({} cycles)", 
                     chunk_idx + 1, num_chunks, start_cycle, actual_end_cycle - 1, chunk_size);
        
        // Load SRAM state from previous chunk (except for first chunk)
        if chunk_idx > 0 {
            sram_storage.copy_from_slice(&sram_state_cpu);
            clilog::info!("Loaded SRAM state from CPU for chunk {}", chunk_idx + 1);
        }
        
        // Create chunk_input_states with cycles_per_chunk + 1 states
        // Structure: [start_cycle_state, (start_cycle+1)_input, (start_cycle+2)_input, ..., end_cycle_input]
        // The simulator will write results to: [start_cycle_output, (start_cycle+1)_output, ..., end_cycle_output]
        let mut chunk_input_states = Vec::new();
        
        // Add the start_cycle state (from previous chunk's final state, or zeros for first chunk)
        chunk_input_states.extend(current_final_state.iter().copied());
        
        // Add VCD input states for cycles (start_cycle+1) to end_cycle
        let vcd_start_offset = (start_cycle + 1) * script.reg_io_state_size as usize;
        let vcd_end_offset = (end_cycle + 1) * script.reg_io_state_size as usize;
        
        // Ensure we don't go out of bounds
        let actual_end_offset = std::cmp::min(vcd_end_offset, input_states.len());
        if vcd_start_offset < actual_end_offset {
            chunk_input_states.extend(&input_states[vcd_start_offset..actual_end_offset]);
        }
        
        // Create chunk_offsets_timestamps by extracting the relevant portion
        let chunk_offsets_timestamps = offsets_timestamps[start_cycle..actual_end_cycle].to_vec();
        
        // Store the length before moving chunk_input_states
        let chunk_input_states_len = chunk_input_states.len();
        
        
        // Transfer only the chunk data to GPU (not the entire input_states)
        let mut input_states_uvec: UVec<_> = chunk_input_states.into();
        input_states_uvec.as_mut_uptr(device);
        
        
        // Run simulation for this chunk
        device.synchronize();
        let timer_sim = clilog::stimer!("simulation_chunk");
        ucci::simulate_v1_noninteractive_simple_scan(
            args.num_blocks,
            script.num_major_stages,
            &script.blocks_start, &script.blocks_data,
            &mut sram_storage,
            chunk_size,
            script.reg_io_state_size as usize,
            &mut input_states_uvec,
            device
        );
        device.synchronize();
        clilog::finish!(timer_sim);

        // Save final state from this chunk (for next chunk initialization)
        if chunk_idx < num_chunks - 1 {
            // Get the final state from this chunk
            // The final state is at chunk_input_states[chunk_size*reg_io_state_size..(chunk_size+1)*reg_io_state_size]
            // This will be used as the start_cycle state for the next chunk
            let final_state_offset = chunk_size * script.reg_io_state_size as usize;
            current_final_state.copy_from_slice(&input_states_uvec[final_state_offset..final_state_offset + script.reg_io_state_size as usize]);
            
            // Save SRAM state to CPU for next chunk
            sram_state_cpu.copy_from_slice(&sram_storage);
            clilog::info!("Saved final state and SRAM state to CPU after chunk {}", chunk_idx + 1);
        }
        
        // Copy results back from GPU to CPU for VCD writing
        let mut chunk_results = vec![0u32; chunk_input_states_len];
        chunk_results.copy_from_slice(&input_states_uvec);
        
        // Write VCD output for this chunk
        clilog::info!("write out vcd");
        write_vcd_chunk(&mut writer, &chunk_results, &chunk_offsets_timestamps, 
                       &out2vcd, script.reg_io_state_size as usize, start_cycle);
        
        clilog::info!("Completed chunk {}/{}", chunk_idx + 1, num_chunks);
    }
    
    clilog::info!("Chunked simulation completed successfully!");
}
