/*
* This file is part of rsrt.
*
* rsrt is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* rsrt is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with rsrt. If not, see <http://www.gnu.org/licenses/>.
*
* Copyright 2015 Chris Foster
*/

use std::borrow::Borrow;
use std::f32;
use std::f32::consts::{PI, FRAC_1_PI};
use std::io::Write;
use std::mem::swap;
use std::ops::{Add, Sub, Mul};

extern crate rand;

#[derive(Copy, Clone, Debug)]
struct Float3 {
	x: f32,
	y: f32,
	z: f32
}

impl Float3 {
	fn new(x: f32, y: f32, z: f32) -> Float3 {
		Float3 { x: x, y: y, z: z}
	}
	
	fn zero() -> Float3 {
		Float3 { x: 0.0, y: 0.0, z: 0.0 }
	}
	
	fn dot(&self, rhs: &Float3) -> f32 {
		self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
	}

	fn cross(&self, rhs: &Float3) -> Float3 {
		Float3 {
			x: self.y * rhs.z - self.z * rhs.y,
			y: self.z * rhs.x - self.x * rhs.z,
			z: self.x * rhs.y - self.y * rhs.x
		}
	}

	fn length(&self) -> f32 {
		self.dot(self).sqrt()
	}

	fn length_squared(&self) -> f32 {
		self.dot(self)
	}

	fn normalized(self) -> Float3 {
		self * (1.0 / self.length())
	}
}

impl Add for Float3 {
	type Output = Float3;

	fn add(self, rhs: Float3) -> Float3 {
		Float3 {
			x: self.x + rhs.x,
			y: self.y + rhs.y,
			z: self.z + rhs.z
		}
	}
}

impl Sub for Float3 {
	type Output = Float3;

	fn sub(self, rhs: Float3) -> Float3 {
		Float3 {
			x: self.x - rhs.x,
			y: self.y - rhs.y,
			z: self.z - rhs.z
		}
	}
}

impl Mul for Float3 {
	type Output = Float3;

	fn mul(self, rhs: Float3) -> Float3 {
		Float3 {
			x: self.x * rhs.x,
			y: self.y * rhs.y,
			z: self.z * rhs.z
		}
	}
}

impl Mul<f32> for Float3 {
	type Output = Float3;

	fn mul(self, rhs: f32) -> Float3 {
		Float3 {
			x: self.x * rhs,
			y: self.y * rhs,
			z: self.z * rhs
		}
	}
}


#[derive(Copy, Clone, Debug)]
struct Ray {
	origin: Float3,
	direction: Float3,
	t_bounds: (f32, f32)
}

impl Ray {
	fn new(origin: Float3, direction: Float3) -> Ray {
		Ray {
			origin: origin,
			direction: direction,
			t_bounds: (0.0, f32::MAX)
		}
	}
}


trait Intersectable {
	fn intersect<'a>(&'a self, ray: &Ray, owner: &'a Object) -> Option<Intersection>;
	//fn intersect_test(&self, ray: &Ray) -> bool;
}

struct Intersection<'a> {
	t: f32,
	normal: Float3,
	//u: f32, v: f32,
	object: &'a Object
}


struct Sphere {
	origin: Float3,
	radius: f32
}

impl Intersectable for Sphere {
	fn intersect<'a>(&'a self, ray: &Ray, owner: &'a Object) -> Option<Intersection> {
		let r = Ray::new(ray.origin - self.origin, ray.direction);
		
		let a = r.direction.length_squared();
		let b = 2.0 * (r.origin.dot(&r.direction));
		let c = r.origin.length_squared() - self.radius * self.radius;
		
		let discriminant = b * b - 4.0 * a * c;
		if discriminant < 0.0 {
			return None;
		}
		
		let discriminant_root = discriminant.sqrt();
		let q = if b < 0.0 {
			-0.5 * (b - discriminant_root)
		} else {
			-0.5 * (b + discriminant_root)
		};
		
		let t0 = &mut (q / a);
		let t1 = &mut (c / q);
		if *t0 > *t1 {
			swap(t0, t1);
		}
		
		if *t0 > ray.t_bounds.1 || *t1 < ray.t_bounds.0 {
			return None;
		}
		
		let t = if *t0 >= ray.t_bounds.0 { *t0 } else { *t1 };
		if t > ray.t_bounds.1 { return None; }
		
		let normal = (r.origin + r.direction * t).normalized();
		
		Some(Intersection { t: t, normal: normal, object: owner })
	}
}


trait BxDF {
	fn sample(&self, wo: &Float3, normal: &Float3) -> (Float3, f32);
	fn pdf(&self, wo: &Float3, normal: &Float3, wi: &Float3) -> f32;
}


struct DiffuseBRDF;

impl BxDF for DiffuseBRDF {
	fn sample(&self, wo: &Float3, normal: &Float3) -> (Float3, f32) {
		let r = rand::random::<f32>().sqrt();
		let theta = 2.0 * PI * rand::random::<f32>();
		
		let x = r * theta.cos();
		let y = r * theta.sin();
		let z = (0.0f32).max(1.0 - x * x - y * y).sqrt();
		
		let tangent = if normal.x != normal.y || normal.x != normal.z {
			Float3::new(normal.z - normal.y, normal.x - normal.z, normal.y - normal.x).normalized()
		} else {
			Float3::new(normal.z - normal.y, normal.x + normal.z, -normal.y - normal.x).normalized()
		};
		
		let bitangent = normal.cross(&tangent);
		
		let wi = tangent * x + bitangent * y + *normal * z;
		
		(wi, z * FRAC_1_PI)
	}
	
	fn pdf(&self, wo: &Float3, normal: &Float3, wi: &Float3) -> f32 {
		(0.0f32).max(normal.dot(wi)) * FRAC_1_PI
	}
}


#[allow(dead_code)]
struct MirrorBRDF;


#[allow(dead_code)]
struct GlassBSDF {
	ior: f32
}


trait Material {
	fn sample(&self, ray: &Ray, i: &Intersection, l: &Float3, throughput: &Float3) -> (Ray, Float3, Float3);
}


struct MatteMaterial {
	color: Float3,
	bsdf: DiffuseBRDF
}

impl MatteMaterial {
	fn new(color: Float3) -> MatteMaterial {
		MatteMaterial { color: color, bsdf: DiffuseBRDF }
	}
}

impl Material for MatteMaterial {
	fn sample(&self, ray: &Ray, i: &Intersection, l: &Float3, throughput: &Float3) -> (Ray, Float3, Float3) {
		let (wi, pdf) = self.bsdf.sample(&ray.direction, &i.normal);
		
		(Ray::new(ray.origin + ray.direction * i.t, wi), *l, self.color * *throughput)
	}
}


struct EmissionMaterial {
	emission: Float3,
	bsdf: DiffuseBRDF
}

impl EmissionMaterial {
	fn new(emission: Float3) -> EmissionMaterial {
		EmissionMaterial { emission: emission, bsdf: DiffuseBRDF }
	}
}

impl Material for EmissionMaterial {
	fn sample(&self, ray: &Ray, i: &Intersection, l: &Float3, throughput: &Float3) -> (Ray, Float3, Float3) {
		let (wi, pdf) = self.bsdf.sample(&ray.direction, &i.normal);
		
		(Ray::new(ray.origin + ray.direction * i.t, wi), *l + *throughput * self.emission, *throughput * self.emission)
	}
}


struct Object {
	intersectable: Box<Intersectable>,
	material: Box<Material>
}

impl Object {
	fn new(intersectable: Box<Intersectable>, material: Box<Material>) -> Object {
		Object {
			intersectable: intersectable,
			material: material
		}
	}
}


struct Scene {
	objects: Vec<Object>
}

impl Scene {
	fn intersect(&self, ray: &Ray) -> Option<Intersection> {
		let mut closest_intersection: Option<Intersection> = None;
		for object in self.objects.iter() {
			match object.intersectable.intersect(ray, object) {
				Some(i) => {
					match closest_intersection {
						Some(i_old) => {
							closest_intersection = Some(
								if i.t < i_old.t { i } else { i_old }
							);
						},
						None => {
							closest_intersection = Some(i);
						}
					}
				},
				None => { }
			}
		}
		
		closest_intersection
	}
}


struct Sample {
	image_x: usize,
	image_y: usize,
}

impl Sample {
	fn new(image_x: usize, image_y: usize) -> Sample {
		Sample { image_x: image_x, image_y: image_y }
	}
}


trait Sampler: Iterator {
	fn get_bounds(&self) -> ((usize, usize), (usize, usize));
	fn get_spp(&self) -> u32;
}


struct SimpleSampler {
	x: usize,
	y: usize,
	x_bounds: (usize, usize),
	y_bounds: (usize, usize),
	s: u32,
	spp: u32
}

impl SimpleSampler {
	fn new(x_bounds: (usize, usize), y_bounds: (usize, usize), spp: u32) -> SimpleSampler {
		SimpleSampler {
			x: x_bounds.0,
			y: y_bounds.0,
			x_bounds: x_bounds,
			y_bounds: y_bounds,
			s: 0,
			spp: spp
		}
	}
}

impl Iterator for SimpleSampler {
	type Item = Sample;
	
	fn next(&mut self) -> Option<Sample> {
		if self.s == self.spp {
			None
		} else {
			let sample = Sample::new(self.x, self.y);
			
			self.x += 1;
			if self.x == self.x_bounds.1 {
				self.x = self.x_bounds.0;
				self.y += 1;
				if self.y == self.y_bounds.1 {
					self.y = self.y_bounds.0;
					self.s += 1;
				}
			}
			
			Some(sample)
		}
	}
}

impl Sampler for SimpleSampler {
	fn get_bounds(&self) -> ((usize, usize), (usize, usize)) {
		(self.x_bounds, self.y_bounds)
	}
	
	fn get_spp(&self) -> u32 {
		self.spp
	}
}


trait Camera {
	fn generate_ray(&self, sample: &Sample) -> Ray;
}


struct PerspectiveCamera {
	look: Ray,
	image_size: (usize, usize)
}

impl PerspectiveCamera {
	fn new(look: Ray, image_size: (usize, usize)) -> PerspectiveCamera {
		PerspectiveCamera { look: look, image_size: image_size }
	}
}

impl Camera for PerspectiveCamera {
	fn generate_ray(&self, sample: &Sample) -> Ray {
		let direction = Float3::new(
			  2.0 * sample.image_x as f32 / self.image_size.0 as f32 - 1.0,
			-(2.0 * sample.image_y as f32 / self.image_size.0 as f32 - 1.0),
			  2.0
		).normalized();
		
		let forward = self.look.direction;
		let right = forward.cross(&Float3::new(0.0, 0.0, 1.0)).normalized();
		let up = right.cross(&forward);
		
		Ray::new(self.look.origin,
			right * direction.x +
			up * direction.y +
			forward * direction.z
		)
	}
}


trait Integrator {
	fn integrate(&self, ray: &Ray) -> Float3;
}


struct PathTracer<'a> {
	scene: &'a Scene
}

impl<'a> Integrator for PathTracer<'a> {
	fn integrate(&self, ray: &Ray) -> Float3 {
		let mut l = Float3::zero();
		let mut throughput = Float3::new(1.0, 1.0, 1.0);
		let mut r = *ray;
		
		loop {
			match self.scene.intersect(&r) {
				Some(i) => {
					let (r2, l2, throughput2) = i.object.material.sample(&r, &i, &l, &throughput);
					
					r = Ray::new(r2.origin + i.normal * 0.001, r2.direction);
					l = l2;
					throughput = throughput2;
				},
				None => { break; }
			}
		}
	
		l
	}
}


fn to_255(value: f32) -> u8 {
	(value.max(0.0).min(1.0) * 255.0 + 0.5) as u8
}


fn main() {
	const IMAGE_WIDTH: usize = 512;
	const IMAGE_HEIGHT: usize = 512;
	const SPP: u32 = 64;

	let scene = Scene {
		objects: vec![
			Object::new(Box::new(Sphere { origin: Float3::new(1.5, 8.0, 0.0), radius: 1.0 }), Box::new(MatteMaterial::new(Float3::new(0.2, 0.9, 0.2)))),
			Object::new(Box::new(Sphere { origin: Float3::new(0.0, 1000.0, 0.0), radius: 990.0 }), Box::new(MatteMaterial::new(Float3::new(0.7, 0.7, 0.7)))),
			Object::new(Box::new(Sphere { origin: Float3::new(-1.5, 8.0, 0.0), radius: 1.0 }), Box::new(EmissionMaterial::new(Float3::new(3.0, 3.0, 3.0))))
		]
	};
	
	let camera = PerspectiveCamera::new(
		Ray::new(Float3::new(0.0, 0.0, 0.0), Float3::new(0.0, 1.0, 0.0)),
		(IMAGE_WIDTH, IMAGE_HEIGHT)
	);
	
	let integrator = PathTracer { scene: &scene };
	
	let sampler = SimpleSampler::new((0, IMAGE_WIDTH), (0, IMAGE_HEIGHT), SPP);
	
	let mut image_buffer = vec![Float3::zero(); IMAGE_WIDTH * IMAGE_HEIGHT];
	
	let sample_scale = 1.0 / SPP as f32;
	for sample in sampler {
		let ray = camera.generate_ray(&sample);
		
		let l = integrator.integrate(&ray);
		
		let pixel = &mut image_buffer[sample.image_y * IMAGE_WIDTH + sample.image_x];
		*pixel = *pixel + l * sample_scale;
	}
	
	let mut image_bytes: Vec<u8> = vec![0; IMAGE_WIDTH * IMAGE_HEIGHT * 3];
	for (index, pixel) in image_buffer.iter().enumerate() {
		image_bytes[index * 3 + 0] = to_255(pixel.x);
		image_bytes[index * 3 + 1] = to_255(pixel.y);
		image_bytes[index * 3 + 2] = to_255(pixel.z);
	}
	
	let image_file_header = format!("P6\n{} {}\n{}\n", IMAGE_WIDTH, IMAGE_HEIGHT, 255);
	
	let image_file_path = std::path::Path::new("image.ppm");
	let mut image_file = match std::fs::File::create(&image_file_path) {
		Err(why) => panic!("Couldn't create image file: {}", why),
		Ok(file) => file
	};

	match image_file.write_all(image_file_header.as_bytes()) {
		Err(why) => panic!("Couldn't write image header: {}", why),
		Ok(_) => { }
	};

	match image_file.write_all(image_bytes.borrow()) {
		Err(why) => panic!("Couldn't write image data: {}", why),
		Ok(_) => { }
	};
}
