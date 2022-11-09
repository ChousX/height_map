use std::ops::{Add, Sub, Mul, Div};

use bevy::{render::mesh::{Indices, Mesh, PrimitiveTopology, VertexAttributeValues}, prelude::Vec3};
use image::{ImageBuffer, Luma, ImageError};
use palette::{Gradient, LinSrgb, Srgb};

pub struct HeightMapBuilder {
    max_height: f32,
    min_height: f32,
    wireframe: bool,
    image: String,
    gradient: Option<Gradient<LinSrgb>>
}

impl HeightMapBuilder{
    pub fn set_image<T: ToString>(mut self, image: T) -> Self{
        self.image = image.to_string();
        self
    }

    pub fn set_max_height(mut self, height: f32) -> Self{
        self.max_height = height;
        self
    }

    pub fn set_min_height(mut self, height: f32) -> Self{
        self.min_height = height;
        self
    }

    pub fn set_gradient(mut self, gradient: Option<Gradient<LinSrgb>>) -> Self{
        self.gradient = gradient;
        self
    }

    pub fn add_gradient(mut self, low: [f32; 3], high: [f32; 3]) -> Self{
        let gradient = Gradient::new(vec![
            LinSrgb::new(low[0], low[1], low[2]),
            LinSrgb::new(high[0], high[1], high[2]),
        ]);

        self.set_gradient(Some(gradient))
    }
}

type PixelData = ImageBuffer<Luma<u16>, Vec<u16>>;
impl HeightMapBuilder{
    fn get_pixel_data(&self) -> Result<PixelData, HeightMapError>{
        let image = match image::open(&self.image){
            Ok(image) => {
                image
            },
            Err(_) => {
                return Err(HeightMapError::ImageOpen)
            }
        };

        match image.as_luma16(){
            Some(height_map) => Ok(height_map.clone()),
            None => Err(HeightMapError::GrayScaleConvertion)
        }
    }
}

impl Default for HeightMapBuilder {
    fn default() -> Self {
        Self {
            max_height: 10.,
            min_height: -10.,
            wireframe: false,
            image: String::new(),
            gradient: None
        }
    }
}

impl HeightMapBuilder {
    pub fn build(&self) -> Result<Mesh, HeightMapError> {
        let pixal_data = self.get_pixel_data()?;
        
        let width = pixal_data.width();
        let height = pixal_data.height();

        let mut vertices: Vec<[f32; 3]> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();
        let mut colors: Vec<[f32; 3]> = Vec::new();
        let mut normals: Vec<[f32; 3]> = Vec::new();
        let mut uvs: Vec<[f32; 2]> = Vec::new();

        for z in 0..height{
            for x in 0..width{
                let pixel = pixal_data.get_pixel(x, z)[0] as f32;
                let y = map_range((0.0, u16::MAX as f32), (self.min_height, self.max_height), pixel);

                let pos = [x as f32, y, z as f32];
                if let Some(gradient) = &self.gradient{
                    let raw_float: Srgb<f32> = Srgb::<f32>::from_linear(gradient.get(y).into());
                    colors.push([raw_float.red, raw_float.green, raw_float.blue]);
                }
                vertices.push(pos);
                uvs.push([0.0, 0.0]);

                debug_assert_eq!(
                    vertices[index(width, x, z) as usize],
                    vertices[vertices.len() - 1]
                )
            }
        }
        for z in 0..height{
            for x in 0..width{
                let mut points: Vec<&[f32; 3]> = Vec::new();

                if x != width - 1 {
                    points.push(&vertices[index(width, x + 1, z) as usize]);
                }

                if z != height - 1 {
                    //println!("{}", index(width, x, z + 1));
                    points.push(&vertices[index(width, x, z + 1) as usize]);
                }

                if x != 0 {
                    points.push(&vertices[index(width, x - 1, z) as usize]);
                }

                if z != 0 {
                    points.push(&vertices[index(width, x, z - 1) as usize]);
                }
    
                let mut sum = Vec3::ZERO;
                for i in 0..points.len() - 1 {
                    sum += surface_normal(
                        &vertices[index(width, x, z) as usize],
                        &points[i],
                        &points[i + 1],
                    );
                }
                sum /= Vec3::splat((points.len() -1) as f32);
                normals.push((-sum.normalize()).into())
            }
        }

        let topology = if self.wireframe {
            //todo generate the indices
            PrimitiveTopology::LineList
        } else {
            for colem in 1..height {
                for row in 1..(width) {
                    let one = index(width, row - 1, colem - 1);
                    let two = index(width, row, colem - 1);
                    let three = index(width, row - 1, colem);
                    let four = index(width, row, colem);
                    indices.append(&mut vec![three, two, one, three, four, two]);
                }
            }
            PrimitiveTopology::TriangleList
        };

        let mut mesh = Mesh::new(topology);
        mesh.insert_attribute(
            Mesh::ATTRIBUTE_POSITION,
            VertexAttributeValues::Float32x3(vertices));
            mesh.set_indices(Some(Indices::U32(indices)));
        mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
        mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);

        Ok(mesh)
    }
}

#[derive(Debug)]
pub enum HeightMapError{
    ImageOpen,
    GrayScaleConvertion
}

#[inline]
fn index(width: u32, x: u32, z: u32) -> u32 {
    z * width + x
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

///from range must enclose the all posiblitlys of s
fn map_range<T: Copy>(from_range: (T, T), to_range: (T, T), s: T) -> T
where
    T: Add<T, Output = T> + Sub<T, Output = T> + Mul<T, Output = T> + Div<T, Output = T>,
{
    to_range.0 + (s - from_range.0) * (to_range.1 - to_range.0) / (from_range.1 - from_range.0)
}