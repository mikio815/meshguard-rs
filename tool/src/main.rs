use anyhow::Result;
use meshguard_quantize::{quantize_positions, encode_normals_oct, quantize_uvs};
use meshguard_pack::pack_interleave_permute;

fn main() -> Result<()> {
    let pos = vec![[0.0, 1.0, 2.0],[10.0,20.0,30.0],[-1.0,0.5,100.0]];
    let nor = vec![[0.0,0.0,1.0],[1.0,0.0,0.0],[0.577,0.577,0.577]];
    let uv  = vec![[0.0,0.0],[0.5,0.75],[1.0,1.0]];
    let idx = vec![0u32, 1, 2];

    let qpos = quantize_positions(&pos);
    let qnor = encode_normals_oct(&nor);
    let quv = quantize_uvs(&uv);

    let packed = pack_interleave_permute(&qpos, &qnor, &quv, Some(&idx), 0xDEAD_BEEF_CAFE_BABE);

    println!("vertex_count: {}", packed.vertex_count);
    println!("interleaved bytes: {}", packed.interleaved.len());
    println!("indices: {:?}", packed.indices);
    println!("pos scale={:?} offset={:?}", packed.pos_scale, packed.pos_offset);
    Ok(())
}