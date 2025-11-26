#[derive(Clone, Copy, Debug)]
pub struct Vec2(pub f32, pub f32);
#[derive(Clone, Copy,Debug)]
pub struct Vec3(pub f32, pub f32, pub f32);

#[derive(Clone, Debug)]
pub struct QuantizedPositions {
    pub data: Vec<i16>,
    pub scale: [f32; 3],
    pub offset: [f32; 3],
}

#[derive(Clone, Debug)]
pub struct QuantizedNormalsOct {
    pub data: Vec<u16>,
}

#[derive(Clone, Debug)]
pub struct QuantizedUVs {
    pub data: Vec<u16>,
}

#[inline]
fn clamp<T: PartialOrd>(x: T, lo: T, hi: T) -> T {
    if x < lo { lo } else if x > hi { hi } else { x }
}

/// AABB: Axis-Aligned Bounding Box
/// モデル全体をちょうど内包する直方体の最小座標と最大座標を返す
pub fn aabb_min_max(positions: &[[f32; 3]]) -> ([f32; 3], [f32; 3]) {
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for p in positions {
        for a in 0..3 {
            if p[a] < min[a] { min[a] = p[a]; }
            if p[a] > max[a] { max[a] = p[a]; }
        }
    }

    for a in 0..3 {
        if (max[a] - min[a]).abs() < 1e-12 {
            max[a] = min[a] + 1e-6;
        }
    }
    (min, max)
}

/// 各頂点の16bit量子化
pub fn quantize_positions(positions: &[[f32; 3]]) -> QuantizedPositions {
    let (min, max) = aabb_min_max(positions);
    let min64 = [min[0] as f64, min[1] as f64, min[2] as f64];
    let rng64 = [
        (max[0] as f64 - min64[0]),
        (max[1] as f64 - min64[1]),
        (max[2] as f64 - min64[2]),
    ];
    let rng64 = [
        if rng64[0].abs() < 1e-12 { 1e-6 } else { rng64[0] },
        if rng64[1].abs() < 1e-12 { 1e-6 } else { rng64[1] },
        if rng64[2].abs() < 1e-12 { 1e-6 } else { rng64[2] },
    ];
    let scale = [
        (rng64[0] / 65535.0) as f32,
        (rng64[1] / 65535.0) as f32,
        (rng64[2] / 65535.0) as f32,
    ];

    let mut data = Vec::with_capacity(positions.len() * 3);
    for p in positions {
        for a in 0..3 {
            let pa = p[a] as f64;
            let t = ((pa - min64[a]) / rng64[a]) * 65535.0;
            let t = t.clamp(0.0, 65535.0);
            let q_u16 = t.round() as i64;
            let q_i32 = (q_u16 - 32768) as i32;
            let q_i32 = q_i32.clamp(-32768, 32767);
            data.push(q_i32 as i16);
        }
    }
    QuantizedPositions { data, scale, offset: min }
}


/// normal: 正規化
/// Octahedral Encoding (https://project-asura.com/blog/archives/11730)
/// 頂点法線を正八面体→平面に展開する
pub fn encode_normals_oct(normals: &[[f32; 3]]) -> QuantizedNormalsOct {
    let mut out = Vec::with_capacity(normals.len() * 2);
    for n in normals {
        let mut x = n[0];
        let mut y = n[1];
        let mut z = n[2];
        let len = (x*x + y*y + z*z).sqrt();
        if len > 0.0 { x /= len; y /= len; z /= len; } else { x = 0.0; y = 0.0; z = 0.0; } // 正規化

        let denom = x.abs() + y.abs() + z.abs();
        let mut px = if denom > 0.0 { x / denom } else { 0.0 };
        let mut py = if denom > 0.0 { y / denom } else { 0.0 };
        if z < 0.0 {
            let sx = px.signum();
            let sy = py.signum();
            let ax = px.abs();
            let ay = py.abs();
            px = (1.0 - ax) * sx;
            py = (1.0 - ay) * sy;
        }

        let u = clamp((px * 0.5 + 0.5) * 65535.0, 0.0, 65535.0).round() as u32;
        let v = clamp((py * 0.5 + 0.5) * 65535.0, 0.0, 65535.0).round() as u32;
        out.push(u as u16);
        out.push(v as u16);
    }
    QuantizedNormalsOct { data: out }
}

pub fn quantize_uvs(uvs: &[[f32; 2]]) -> QuantizedUVs {
    let mut out = Vec::with_capacity(uvs.len() * 2);
    for uv in uvs {
        let u = clamp(uv[0], 0.0, 1.0);
        let v = clamp(uv[1], 0.0, 1.0);
        let u = (u * 65535.0).round() as u32;
        let v = (v * 65535.0).round() as u32;
        out.push(u as u16);
        out.push(v as u16);
    }
    QuantizedUVs { data: out }
}

/// テスト復号用
pub fn dequantize_positions(q: &QuantizedPositions) -> Vec<[f32; 3]> {
    let scale64 = [q.scale[0] as f64, q.scale[1] as f64, q.scale[2] as f64];
    let off64   = [q.offset[0] as f64, q.offset[1] as f64, q.offset[2] as f64];
    let mut out = Vec::with_capacity(q.data.len() / 3);
    for i in 0..(q.data.len() / 3) {
        let xq = q.data[i*3 + 0] as i32 as f64;
        let yq = q.data[i*3 + 1] as i32 as f64;
        let zq = q.data[i*3 + 2] as i32 as f64;
        let x = ((xq + 32768.0) * scale64[0]) + off64[0];
        let y = ((yq + 32768.0) * scale64[1]) + off64[1];
        let z = ((zq + 32768.0) * scale64[2]) + off64[2];
        out.push([x as f32, y as f32, z as f32]);
    }
    out
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_positions() {
        let src = vec![[0.0, 1.0, 2.0], [10.0, 20.0, 30.0], [-1.0, 0.5, 100.0]];
        let q = quantize_positions(&src);
        let restored = dequantize_positions(&q);
        assert_eq!(restored.len(), src.len());
        for (i, p) in src.iter().enumerate() {
            for a in 0..3 {
                let diff = (p[a] - restored[i][a]).abs();
                let tol = 0.5 * q.scale[a] + 1e-6;
                assert!(diff <= tol, "diff {} exceeds tol {}", diff, tol);
            }
        }
    }
}