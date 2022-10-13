use std::ops::{Add, Div, Mul, Neg, Sub};
use std::path::{Path, PathBuf};

use bevy::prelude::*;

use bevy::render::mesh::{Indices, PrimitiveTopology, VertexAttributeValues};
use image::{ImageBuffer, Luma};
use palette::{Gradient, LinSrgb, Srgb};

pub type HeightMapU16 = ImageBuffer<Luma<u16>, Vec<u16>>;

#[derive(Default, Clone)]
pub struct HeightMapBuilder {
    path: Option<PathBuf>,
    wireframe: bool,
    min_height: Option<f32>,
    max_height: Option<f32>,
}

impl HeightMapBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_input<P>(&mut self, input: P) -> &mut Self
    where
        P: AsRef<Path>,
    {
        self.path = Some(input.as_ref().into());
        self
    }

    pub fn set_wierframe(&mut self, input: bool) -> &mut Self {
        self.wireframe = input;
        self
    }

    pub fn set_max_height(&mut self, height: f32) -> &mut Self {
        self.max_height = Some(height);
        self
    }

    pub fn set_min_height(&mut self, height: f32) -> &mut Self {
        self.min_height = Some(height);
        self
    }

    pub fn build(&mut self) -> Mesh {
        let path = self
            .path
            .clone()
            .expect("You need to set a path befor running HeightMapBuilder");
        let terrain_image = image::open(path).expect("failed to open terrain.png");

        let height_map = terrain_image
            .as_luma16()
            .expect("failed to convert terrain_image into heightmap");

        let wireframe = self.wireframe;
        let max = self.max_height.unwrap_or(100.);
        let min = self.min_height.unwrap_or(0.);

        let width = height_map.width();
        let langth = height_map.height();

        let gradient = Gradient::new(vec![
            LinSrgb::new(1.0, 1.0, 0.0),
            LinSrgb::new(0.0, 0.0, 1.0),
        ]);

        let mut vertices: Vec<[f32; 3]> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();
        let mut normals: Vec<[f32; 3]> = Vec::new();
        let mut uvs: Vec<[f32; 2]> = Vec::new();
        let mut colors: Vec<[f32; 3]> = Vec::new();

        //      y   y height
        //      | z | z langth
        //      |/  |/
        //  x---/   /---x width

        for z in 0..langth {
            for x in 0..width {
                let pixel = (height_map.get_pixel(x, z).0)[0] as f32;
                let y = map_range((0.0, u16::MAX as f32), (min, max), pixel);
                let pos = [x as f32, y, z as f32];
                let color = gradient.get(y);
                let raw_float: Srgb<f32> = Srgb::<f32>::from_linear(color.into());

                vertices.push(pos);
                colors.push([raw_float.red, raw_float.green, raw_float.blue]);
                uvs.push([0.0, 0.0]);
            }
        }

        for z in 0..langth {
            for x in 0..width {
                let mut points: Vec<&[f32; 3]> = Vec::new();
                if x != width - 1 {
                    points.push(&vertices[index(width, x + 1, z)]);
                }
                if z != langth - 1 {
                    //println!("{}", index(width, x, z + 1));
                    points.push(&vertices[index(width, x, z + 1)]);
                }
                if x != 0 {
                    points.push(&vertices[index(width, x - 1, z)]);
                }
                if z != 0 {
                    points.push(&vertices[index(width, x, z - 1)]);
                }

                let mut sum = Vec3::ZERO;
                let num = points.len() - 1;
                for i in 0..num - 1 {
                    sum +=
                        surface_normal(&vertices[index(width, x, z)], &points[i], &points[i + 1]);
                }
                let num = num as f32;

                sum /= Vec3::new(num, num, num);
                normals.push(sum.normalize().neg().into())
            }
        }

        let mut mesh = if wireframe {
            Mesh::new(PrimitiveTopology::LineList)
        } else {
            Mesh::new(PrimitiveTopology::TriangleList)
        };

        let colors: Vec<[f32; 4]> = colors
            .into_iter()
            .map(|color| [color[0], color[1], color[2], 0.5])
            .collect();

        mesh.insert_attribute(
            Mesh::ATTRIBUTE_COLOR,
            VertexAttributeValues::Float32x4(colors.into()),
        );

        mesh.insert_attribute(
            Mesh::ATTRIBUTE_POSITION,
            VertexAttributeValues::Float32x3(vertices),
        );
        mesh.set_indices(Some(Indices::U32(indices)));
        mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
        mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);

        mesh
    }
}

#[inline]
fn index(width: u32, x: u32, z: u32) -> usize {
    let (x, z, w) = (x as usize, z as usize, width as usize);
    z * w + x
}

#[inline]
fn map_range<T: Copy>(from_range: (T, T), to_range: (T, T), s: T) -> T
where
    T: Add<T, Output = T> + Sub<T, Output = T> + Mul<T, Output = T> + Div<T, Output = T>,
{
    to_range.0 + (s - from_range.0) * (to_range.1 - to_range.0) / (from_range.1 - from_range.0)
}

#[inline]
fn surface_normal(p1: &[f32; 3], p2: &[f32; 3], p3: &[f32; 3]) -> Vec3 {
    let p1: Vec3 = p1.clone().into();
    let p2: Vec3 = p2.clone().into();
    let p3: Vec3 = p3.clone().into();

    let u = p2 - p1;
    let v = p3 - p1;

    let normal = Vec3::new(
        u.y * v.z - u.z * v.y,
        u.z * v.x - u.x * v.z,
        u.x * v.y - u.y * v.x,
    );

    normal
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn has_all_the_data() {
        let mut builder = HeightMapBuilder::new();
        builder
            .set_input("terrain.png")
            .set_wierframe(false)
            .set_max_height(50.)
            .set_min_height(-50.)
            .build();
    }
}
