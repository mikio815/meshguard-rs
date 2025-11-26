use meshguard_quantize::{QuantizedPositions, QuantizedNormalsOct, QuantizedUVs};

#[derive(Clone, Debug)]
pub struct PackedMesh {
    pub interleaved: Vec<u8>,
    pub vertex_count: usize,
    pub indices: Vec<u32>,
    pub pos_scale: [f32; 3],
    pub pos_offset: [f32; 3],
    pub perm_seed: u64,
}

/// ランダム順列作るだけ（シード保存用）
fn permutation_fy(n: usize, seed: u64) -> Vec<u32> {
    let mut p: Vec<u32> = (0..n as u32).collect();
    let mut s = if seed == 0 { 0x9E3779B97F4A7C15 } else { seed };
    let mut next_u64 = || {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        s.wrapping_mul(0x2545F4914F6CDD1D)
    };
    for i in (1..n).rev() {
        let r = next_u64() as usize % (i + 1);
        p.swap(i, r);
    }
    p
}

/// 逆写像
fn inverse_permutation(perm: &[u32]) -> Vec<u32> {
    let mut inv = vec![0u32; perm.len()];
    for (new, &old) in perm.iter().enumerate() {
        inv[old as usize] = new as u32;
    }
    inv
}

#[inline] fn push_i16_le(buf: &mut Vec<u8>, v: i16) { buf.extend_from_slice(&v.to_le_bytes()); }
#[inline] fn push_u16_le(buf: &mut Vec<u8>, v: u16) { buf.extend_from_slice(&v.to_le_bytes()); }

pub fn pack_interleave_permute(
    qpos: &QuantizedPositions,
    qnor: &QuantizedNormalsOct,
    quv:  &QuantizedUVs,
    indices: Option<&[u32]>,
    perm_seed: u64,
) -> PackedMesh {
    let vertex_count = qpos.data.len() / 3;
    assert_eq!(qnor.data.len(), vertex_count * 2, "normal length mismatch");
    assert_eq!(quv.data.len(),  vertex_count * 2, "uv length mismatch");

    let perm = permutation_fy(vertex_count, perm_seed);
    let inv  = inverse_permutation(&perm);

    let mut interleaved = Vec::with_capacity(vertex_count * (3*2 + 2*2 + 2*2));
    for new_idx in 0..vertex_count {
        let old_idx = perm[new_idx] as usize;

        let px = qpos.data[old_idx * 3 + 0];
        let py = qpos.data[old_idx * 3 + 1];
        let pz = qpos.data[old_idx * 3 + 2];
        push_i16_le(&mut interleaved, px);
        push_i16_le(&mut interleaved, py);
        push_i16_le(&mut interleaved, pz);

        let nu = qnor.data[old_idx * 2 + 0];
        let nv = qnor.data[old_idx * 2 + 1];
        push_u16_le(&mut interleaved, nu);
        push_u16_le(&mut interleaved, nv);

        let uu = quv.data[old_idx * 2 + 0];
        let vv = quv.data[old_idx * 2 + 1];
        push_u16_le(&mut interleaved, uu);
        push_u16_le(&mut interleaved, vv);
    }

    // 逆写像を元に壊れたインデックスを治す必要がある
    let remapped_indices = if let Some(idx) = indices {
        let mut out = Vec::with_capacity(idx.len());
        for &i in idx {
            let newi = inv[i as usize];
            out.push(newi);
        }
        out
    } else {
        (0..vertex_count as u32).collect()
    };

    PackedMesh {
        interleaved,
        vertex_count,
        indices: remapped_indices,
        pos_scale: qpos.scale,
        pos_offset: qpos.offset,
        perm_seed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use meshguard_quantize::{quantize_positions, encode_normals_oct, quantize_uvs};

    #[test]
    fn pack_round_lengths() {
        let pos = vec![[0.0,1.0,2.0],[10.0,20.0,30.0],[-1.0,0.5,100.0]];
        let nor = vec![[0.0,0.0,1.0],[1.0,0.0,0.0],[0.577,0.577,0.577]];
        let uv  = vec![[0.0,0.0],[0.5,0.75],[1.0,1.0]];
        let qpos = quantize_positions(&pos);
        let qnor = encode_normals_oct(&nor);
        let quv  = quantize_uvs(&uv);

        let idx  = vec![0u32, 1, 2];
        let packed = pack_interleave_permute(&qpos, &qnor, &quv, Some(&idx), 0x1234_5678_9ABC_DEF0);

        assert_eq!(packed.vertex_count, 3);
        assert_eq!(packed.interleaved.len(), 3 * 14);
        assert_eq!(packed.indices.len(), 3);
    }
}
