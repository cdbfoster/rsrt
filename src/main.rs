// This file is part of rsrt.
//
// rsrt is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// rsrt is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with rsrt. If not, see <http://www.gnu.org/licenses/>.
//
// Copyright 2018 Chris Foster

//! `rsrt` is a small, but very extensible ray tracing framework with some useful implementations.  It is useable, and provides a basic
//! forward path tracer, a sphere, and some simple materials.
//!
//! To build and render the default scene, use
//! ```
//! $ cargo run --release
//! ```
//! To build and view the documentation, use
//! ```
//! $ cargo doc --open
//! ```
//!
//! While it is possible to use `rsrt`'s traits and implementations in other projects, it is not intended to be more than its simple example
//! binary; its modules are marked `pub` so that `rustdoc` picks them up.  If you want to use them elsewhere, you can move what you need
//! into a separate `lib.rs` file and have `cargo` build you an actual library.

// This is our only external dependency
extern crate rand;

const IMAGE_WIDTH: u32 = 512;
const IMAGE_HEIGHT: u32 = 512;
const SPP: u32 = 64;

/// Sample program that renders a small scene and outputs a .ppm file.
pub fn main() {
    use camera::impls::PerspectiveCamera;
    use integration::impls::PathTracingIntegrator;
    use math::{Float3, Ray};
    use sampling::impls::SimpleSampler;
    use scene::{Object, Scene};
    use shading::impls::{EmissionShader, MatteShader, MirrorShader};
    use shapes::Sphere;

    // Setup a few spheres as the scene
    let scene = Scene::new(vec![
        Box::new(Object::new(
            Sphere::new(
                Float3::new(1.5, 8.0, 0.0),  // origin
                1.0,                         // radius
            ), MatteShader::new(
                Float3::new(0.2, 0.9, 0.2),  // color
            ),
        )),
        Box::new(Object::new(
            Sphere::new(
                Float3::new(0.0, 0.0, 0.0),  // origin
                11.0,                        // radius
            ), MatteShader::new(
                Float3::new(0.7, 0.7, 0.7),  // color
            ),
        )),
        Box::new(Object::new(
            Sphere::new(
                Float3::new(-1.5, 8.0, 0.0), // origin
                1.0,                         // radius
            ), EmissionShader::new(
                Float3::new(5.0, 5.0, 5.0),  // emission
            ),
        )),
        Box::new(Object::new(
            Sphere::new(
                Float3::new(0.0, 9.5, 0.0),  // origin
                1.0,                         // radius
            ), MirrorShader::new(
                Float3::new(0.6, 0.6, 0.6),  // color
            ),
        )),
    ]);

    // This object will be helping us to solve the rendering integral
    let integrator = PathTracingIntegrator::new(&scene);

    // Generates samples for us, which become the starting points for rays to cast into the scene
    let sampler = SimpleSampler::new((0, IMAGE_WIDTH), (0, IMAGE_HEIGHT), SPP);

    // Turns samples into rays
    let camera = PerspectiveCamera::new(
        Ray::new(
            Float3::new(0.0, 4.0, 5.0),               // position
            Float3::new(0.0, 4.5, -5.0).normalized(), // look
        ),
        (IMAGE_WIDTH, IMAGE_HEIGHT),
    );

    // Render!
    let image_buffer_f32 = render(&integrator, &sampler, &camera);
    let image_buffer_u8 = clamp_and_quantize(&image_buffer_f32);
    write_ppm_file("image.ppm", &image_buffer_u8);
}

/// Accepts an image buffer of floating point data, clamps each channel to [0, 1], and
/// quantizes each channel to 8 bits.
pub fn clamp_and_quantize(image_buffer: &Vec<math::Float3>) -> Vec<u8> {
    let mut bytes: Vec<u8> = vec![0; (IMAGE_WIDTH * IMAGE_HEIGHT * 3) as usize];

    for (index, pixel) in image_buffer.iter().enumerate() {
        bytes[index * 3 + 0] = math::to_255(pixel.x);
        bytes[index * 3 + 1] = math::to_255(pixel.y);
        bytes[index * 3 + 2] = math::to_255(pixel.z);
    }

    bytes
}

/// Uses the integrator to evaluate the integral along as many rays as the given sampler provides.
pub fn render<'a, I, S, C>(
    integrator: &I,
    sampler: &'a S,
    camera: &C,
) -> Vec<math::Float3> where
    I: integration::Integrator,
    S: sampling::Sampler<'a>,
    C: camera::Camera,
{
    let mut image_buffer = vec![math::Float3::zero(); (IMAGE_WIDTH * IMAGE_HEIGHT) as usize];

    println!("Rendering...");
    let sample_scale = 1.0 / SPP as f32;
    for (i, sample) in sampler.iter().enumerate() {
        let ray = camera.generate_ray(&sample);

        let radiance = integrator.integrate(ray);

        let pixel = &mut image_buffer[(sample.image_y * IMAGE_WIDTH + sample.image_x) as usize];
        *pixel = *pixel + radiance * sample_scale;

        if (i + 1) % (IMAGE_WIDTH * IMAGE_HEIGHT) as usize == 0 {
            let current_spp = (i + 1) / (IMAGE_WIDTH * IMAGE_HEIGHT) as usize;
            println!("  {} / {} spp, {:.2}%", current_spp, SPP, current_spp as f32 / SPP as f32 * 100.0);
        }
    }
    println!("Render complete.");

    image_buffer
}

/// Writes a quantized image buffer into a .ppm file.
pub fn write_ppm_file(filename: &str, image_buffer: &Vec<u8>) {
    use std::io::Write;

    println!("Writing file...");
    let image_file_path = std::path::Path::new(filename);
    let mut image_file = match std::fs::File::create(&image_file_path) {
        Err(error) => panic!("Couldn't create image file: {}", error),
        Ok(file) => file,
    };

    let image_file_header = format!("P6\n{} {}\n{}\n", IMAGE_WIDTH, IMAGE_HEIGHT, 255);
    if let Err(error) = image_file.write_all(image_file_header.as_bytes()) {
        panic!("Couldn't write image header: {}", error);
    }

    if let Err(error) = image_file.write_all(image_buffer) {
        panic!("Couldn't write image data: {}", error);
    }
    println!("Wrote file \"{}\".", filename);
}

pub mod bsdf {
    //! Bidirectional Scattering Distribution Functions -
    //! Trait and implementations that define various light-scattering behaviors.

    use math::Float3;

    /// Methods for describing a surface's light-scattering properties.
    pub trait Bsdf {
        /// Returns a random incoming vector for the given outgoing vector, and the pdf.
        fn sample(&self, outgoing: Float3, normal: Float3) -> BsdfSample;
        /// Unused, so far.
        fn pdf(&self, outgoing: Float3, normal: Float3, incoming: Float3) -> f32;
    }

    pub struct BsdfSample {
        pub incoming: Float3,
        pub pdf: f32,
    }

    pub mod impls {
        //! BSDF implementations
        use std::f32::consts::{FRAC_1_PI, PI};

        use math::Float3;
        use bsdf::{Bsdf, BsdfSample};

        /// BRDF for perfect diffuse scattering.
        #[derive(Debug)]
        pub struct DiffuseBrdf;

        impl Bsdf for DiffuseBrdf {
            fn sample(&self, _: Float3, normal: Float3) -> BsdfSample {
                use rand;

                let r = rand::random::<f32>().sqrt();
                let theta = 2.0 * PI * rand::random::<f32>();

                let x = r * theta.cos();
                let y = r * theta.sin();
                let z = (0.0f32).max(1.0 - x * x - y * y).sqrt();

                let (tangent, bitangent) = normal.make_orthonormals();
                let incoming = tangent * x + bitangent * y + normal * z;

                BsdfSample { incoming: incoming, pdf: z * FRAC_1_PI }
            }

            fn pdf(&self, _: Float3, normal: Float3, incoming: Float3) -> f32 {
                (0.0f32).max(normal.dot(&incoming)) * FRAC_1_PI
            }
        }

        /// BRDF for perfect specular scattering.
        #[derive(Debug)]
        pub struct MirrorBrdf;

        impl Bsdf for MirrorBrdf {
            fn sample(&self, outgoing: Float3, normal: Float3) -> BsdfSample {
                let incoming = normal * 2.0 * normal.dot(&outgoing) - outgoing;
                BsdfSample { incoming: incoming, pdf: 1.0 }
            }

            fn pdf(&self, _: Float3, _: Float3, _: Float3) -> f32 {
                0.0
            }
        }
    }
}

pub mod camera {
    //! Cameras are used to generate [rays](math/Ray.t.html) from [samples](sampling/Sample.t.html).

    use math::Ray;
    use sampling::Sample;

    pub trait Camera {
        /// Generates a [ray](math/Ray.t.html) from a [sample](sampling/Sample.t.html).
        fn generate_ray(&self, sample: &Sample) -> Ray;
    }

    pub mod impls {
        use camera::Camera;
        use math::{Float3, Ray};
        use sampling::Sample;

        /// Generates rays from a single point in space, with perspective.
        /// Currently rays are projected from the camera's origin to an image grid positioned 2 units away along the camera's look axis.
        #[derive(Debug)]
        pub struct PerspectiveCamera {
            look: Ray,
            image_size: (u32, u32),
        }

        impl PerspectiveCamera {
            /// Creates a camera centered at `look`'s origin, and oriented `look`'s direction.
            /// `image_size` describes the size of the grid through which to project samples.
            pub fn new(look: Ray, image_size: (u32, u32)) -> Self {
                Self {
                    look: look,
                    image_size: image_size,
                }
            }
        }

        impl Camera for PerspectiveCamera {
            fn generate_ray(&self, sample: &Sample) -> Ray {
                let direction = Float3::new(2.0 * sample.image_x as f32 / self.image_size.0 as f32 - 1.0,
                                          -(2.0 * sample.image_y as f32 / self.image_size.0 as f32 - 1.0),
                                            2.0)
                                    .normalized();

                let forward = self.look.direction;
                let right = forward.cross(&Float3::new(0.0, 0.0, 1.0)).normalized();
                let up = right.cross(&forward);

                Ray::new(
                    self.look.origin,
                    right * direction.x + up * direction.y + forward * direction.z,
                )
            }
        }
    }
}

pub mod integration {
    //! Trait and implementations that attempt to evaluate the integral of the rendering equation.

    use math::{Float3, Ray};

    pub trait Integrator {
        /// Integrate along the given ray, returning an estimate of the incoming radiance.
        fn integrate(&self, ray: Ray) -> Float3;
    }

    pub mod impls {
        //! Integrator implementations

        use rand;

        use integration::Integrator;
        use math::{Float3, Ray};
        use scene::Scene;
        use shading::{Shadeable, ShaderSample};

        /// A simple forward path tracer.  Rays bounce a minimum of 3 times, and then are subjected to Russian Roulette termination.
        #[derive(Debug)]
        pub struct PathTracingIntegrator<'a> {
            scene: &'a Scene,
        }

        impl<'a> PathTracingIntegrator<'a> {
            pub fn new(scene: &'a Scene) -> Self {
                Self {
                    scene: scene,
                }
            }
        }

        impl<'a> Integrator for PathTracingIntegrator<'a> {
            fn integrate(&self, mut ray: Ray) -> Float3 {
                let mut radiance = Float3::zero();
                let mut throughput = Float3::new(1.0, 1.0, 1.0);

                let mut bounces = 0;
                'cast_ray: loop {
                    if let Some(mut point) = self.scene.cast_ray(&ray) {
                        // If we've hit the backside of a surface, flip the normal for the sake of our calculations
                        if ray.direction.dot(&point.intersection.normal) > 0.0 {
                            point.intersection.normal = point.intersection.normal * -1.0;
                        }

                        // Sample a random bounce off the surface
                        let ShaderSample {
                            outgoing: next_ray,
                            radiance: next_radiance,
                            throughput: next_throughput,
                        } = point.shader.sample(&ray, &point.intersection, radiance, throughput);

                        // Nudge the next ray slightly off the surface to prevent autocollisions
                        ray = Ray::new(next_ray.origin + point.intersection.normal * 0.001, next_ray.direction);
                        radiance = next_radiance;
                        throughput = next_throughput;

                        bounces += 1;
                        // If we've bounced more than three times, terminate rays with random chance
                        if bounces > 3 {
                            let probability = throughput.average();

                            if probability == 0.0 {
                                break 'cast_ray;
                            } else {
                                if probability < rand::random::<f32>() {
                                    break 'cast_ray;
                                }

                                throughput = throughput * probability;
                            }
                        }
                    } else {
                        break 'cast_ray;
                    }
                }

                // The accumulated radiance is what we're after
                radiance
            }
        }
    }
}

pub mod intersection {
    //! Trait and struct for detecting/describing ray-surface intersections.
    use std::fmt::Debug;

    use math::{Float3, Ray};

    /// A surface that can be detected by a ray.
    pub trait Intersectable: Debug {
        fn intersect<'a>(&'a self, ray: &Ray) -> Option<Intersection>;
    }

    /// An intersection of a surface by a ray.
    #[derive(Debug)]
    pub struct Intersection<'a> {
        /// The time along the ray at which the intersection occurred.
        pub t: f32,
        /// The normal of the surface at the intersection point.
        pub normal: Float3,
        // u: f32, v: f32,
        /// The surface that was intersected.
        pub intersectable: &'a Intersectable,
    }
}

pub mod math {
    //! A couple geometry-related structures and common operations.
    use std::f32;
    use std::ops::{Add, Sub, Mul, Neg};

    /// A vector of 3 floating point numbers.
    #[derive(Clone, Copy, Debug)]
    pub struct Float3 {
        pub x: f32,
        pub y: f32,
        pub z: f32,
    }

    impl Float3 {
        pub fn new(x: f32, y: f32, z: f32) -> Self {
            Self { x: x, y: y, z: z }
        }

        pub fn zero() -> Self {
            Self {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            }
        }

        /// Vector dot product.
        pub fn dot(&self, rhs: &Self) -> f32 {
            self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
        }

        /// Vector cross product.
        pub fn cross(&self, rhs: &Self) -> Self {
            Self {
                x: self.y * rhs.z - self.z * rhs.y,
                y: self.z * rhs.x - self.x * rhs.z,
                z: self.x * rhs.y - self.y * rhs.x,
            }
        }

        pub fn length(&self) -> f32 {
            self.dot(self).sqrt()
        }

        pub fn length_squared(&self) -> f32 {
            self.dot(self)
        }

        pub fn normalized(self) -> Self {
            self * (1.0 / self.length())
        }

        /// Returns a tangent and bitangent vector
        pub fn make_orthonormals(&self) -> (Self, Self) {
            let tangent = if self.x != self.y || self.x != self.z {
                Self::new(self.z - self.y, self.x - self.z, self.y - self.x).normalized()
            } else {
                Self::new(self.z - self.y, self.x + self.z, -self.y - self.x).normalized()
            };

            let bitangent = self.cross(&tangent);

            (tangent, bitangent)
        }

        /// Returns the average of the vector's elements
        pub fn average(&self) -> f32 {
            (self.x + self.y + self.z) / 3.0
        }
    }

    impl Add for Float3 {
        type Output = Float3;

        fn add(self, rhs: Self) -> Self {
            Self {
                x: self.x + rhs.x,
                y: self.y + rhs.y,
                z: self.z + rhs.z,
            }
        }
    }

    impl Sub for Float3 {
        type Output = Float3;

        fn sub(self, rhs: Self) -> Self {
            Self {
                x: self.x - rhs.x,
                y: self.y - rhs.y,
                z: self.z - rhs.z,
            }
        }
    }

    impl Mul for Float3 {
        type Output = Float3;

        /// This is an element-wise multiplication.
        fn mul(self, rhs: Self) -> Self {
            Self {
                x: self.x * rhs.x,
                y: self.y * rhs.y,
                z: self.z * rhs.z,
            }
        }
    }

    impl Mul<f32> for Float3 {
        type Output = Float3;

        fn mul(self, rhs: f32) -> Self {
            Self {
                x: self.x * rhs,
                y: self.y * rhs,
                z: self.z * rhs,
            }
        }
    }

    impl Neg for Float3 {
        type Output = Float3;

        fn neg(self) -> Self {
            Self {
                x: -self.x,
                y: -self.y,
                z: -self.z,
            }
        }
    }

    /// A ray consisting of an origin and direction, along with bounds on length.
    #[derive(Clone, Copy, Debug)]
    pub struct Ray {
        pub origin: Float3,
        pub direction: Float3,
        pub t_bounds: (f32, f32),
    }

    impl Ray {
        pub fn new(origin: Float3, direction: Float3) -> Self {
            Self {
                origin: origin,
                direction: direction,
                t_bounds: (0.0, f32::MAX),
            }
        }
    }

    /// Clamp an f32 value to [0, 1] and quantize to 8 bits.
    pub fn to_255(value: f32) -> u8 {
        (value.max(0.0).min(1.0) * 255.0 + 0.5) as u8
    }
}

pub mod sampling {
    //! Trait and implementations for generating image locations to sample.

    /// An image sample.
    pub struct Sample {
        pub image_x: u32,
        pub image_y: u32,
    }

    impl Sample {
        pub fn new(image_x: u32, image_y: u32) -> Self {
            Self {
                image_x: image_x,
                image_y: image_y,
            }
        }
    }

    /// A sample generator.
    pub trait Sampler<'a> {
        type Iter: Iterator<Item = Sample>;

        /// Returns an iterator than can be used to cycle through the samples generator by this sampler.
        fn iter(&'a self) -> Self::Iter;
        /// Returns the image rectangle in which this sampler will generate samples.
        /// The return value is in the form of `((min_x, max_x), (min_y, max_y))`.
        fn get_bounds(&self) -> ((u32, u32), (u32, u32));
        /// Returns the average samples per pixel this sampler will generate.
        fn get_total_spp(&self) -> u32;
    }

    pub mod impls {
        //! Sampler implementations

        use sampling::{Sample, Sampler};

        /// Simply generates a sample per pixel, sequentially until the desired samples per pixel have been reached.
        pub struct SimpleSampler {
            x_bounds: (u32, u32),
            y_bounds: (u32, u32),
            spp: u32,
        }

        impl SimpleSampler {
            pub fn new(x_bounds: (u32, u32), y_bounds: (u32, u32), spp: u32) -> Self {
                Self {
                    x_bounds: x_bounds,
                    y_bounds: y_bounds,
                    spp: spp,
                }
            }
        }

        impl<'a> Sampler<'a> for SimpleSampler {
            type Iter = SimpleSamplerIterator<'a>;

            fn iter(&'a self) -> Self::Iter {
                Self::Iter::new(self)
            }

            fn get_bounds(&self) -> ((u32, u32), (u32, u32)) {
                (self.x_bounds, self.y_bounds)
            }

            fn get_total_spp(&self) -> u32 {
                self.spp
            }
        }

        /// An iterator over a [SimpleSampler](SimpleSampler.t.html).
        pub struct SimpleSamplerIterator<'a> {
            x: u32,
            y: u32,
            s: u32,
            sampler: &'a SimpleSampler,
        }

        impl<'a> SimpleSamplerIterator<'a> {
            pub fn new(sampler: &'a SimpleSampler) -> Self {
                Self {
                    x: 0,
                    y: 0,
                    s: 0,
                    sampler: sampler,
                }
            }
        }

        impl<'a> Iterator for SimpleSamplerIterator<'a> {
            type Item = Sample;

            fn next(&mut self) -> Option<Sample> {
                if self.s == self.sampler.spp {
                    None
                } else {
                    let sample = Sample::new(self.x, self.y);

                    self.x += 1;
                    if self.x == self.sampler.x_bounds.1 {
                        self.x = self.sampler.x_bounds.0;
                        self.y += 1;
                        if self.y == self.sampler.y_bounds.1 {
                            self.y = self.sampler.y_bounds.0;
                            self.s += 1;
                        }
                    }

                    Some(sample)
                }
            }
        }
    }
}

pub mod scene {
    //! Scene and object definitions.

    use intersection::Intersectable;
    use math::Ray;
    use shading::{Shadeable, Shader, ShadingPoint};

    /// The combination of a shader with an intersectable surface.
    #[derive(Debug)]
    pub struct Object<I: Intersectable, S: Shader> {
        intersectable: I,
        shader: S,
    }

    impl<I: Intersectable, S: Shader> Object<I, S> {
        pub fn new(intersectable: I, shader: S) -> Self {
            Self {
                intersectable: intersectable,
                shader: shader,
            }
        }
    }

    impl<I: Intersectable, S: Shader> Shadeable for Object<I, S> {
        fn cast_ray(&self, ray: &Ray) -> Option<ShadingPoint> {
            if let Some(intersection) = self.intersectable.intersect(ray) {
                Some(ShadingPoint {
                    intersection: intersection,
                    shader: &self.shader,
                })
            } else {
                None
            }
        }
    }

    /// A collection of shadeable objects.
    #[derive(Debug)]
    pub struct Scene {
        pub objects: Vec<Box<Shadeable>>,
    }

    impl Scene {
        pub fn new(objects: Vec<Box<Shadeable>>) -> Self {
            Self {
                objects: objects,
            }
        }
    }

    impl Shadeable for Scene {
        fn cast_ray(&self, ray: &Ray) -> Option<ShadingPoint> {
            let mut closest_point: Option<ShadingPoint> = None;

            for object in &self.objects {
                if let Some(point) = object.cast_ray(ray) {
                    if let Some(previous_closest) = closest_point {
                        if point.intersection.t < previous_closest.intersection.t {
                            closest_point = Some(point);
                        } else {
                            closest_point = Some(previous_closest);
                        }
                    } else {
                        closest_point = Some(point);
                    }
                }
            }

            closest_point
        }
    }
}

pub mod shading {
    //! Traits and implementations for describing material properties and the interaction between rays and materials.
    use std::fmt::Debug;

    use intersection::Intersection;
    use math::{Float3, Ray};

    /// An object that can be detected by a ray and has material properties.
    pub trait Shadeable: Debug {
        fn cast_ray(&self, ray: &Ray) -> Option<ShadingPoint>;
    }

    /// A material.
    pub trait Shader: Debug {
        /// Returns a new ray after bouncing the old one off of the material, along with
        /// the new radiance and throughput.
        fn sample(&self, incoming: &Ray, intersection: &Intersection, radiance: Float3, throughput: Float3) -> ShaderSample;
    }

    pub struct ShaderSample {
        pub outgoing: Ray,
        pub radiance: Float3,
        pub throughput: Float3,
    }

    pub struct ShadingPoint<'a> {
        pub intersection: Intersection<'a>,
        pub shader: &'a Shader,
    }

    pub mod impls {
        //! Shader implementations

        use bsdf::{Bsdf, BsdfSample};
        use bsdf::impls::{DiffuseBrdf, MirrorBrdf};
        use intersection::Intersection;
        use math::{Float3, Ray};
        use shading::{Shader, ShaderSample};

        /// A material that emits light.
        #[derive(Debug)]
        pub struct EmissionShader {
            emission: Float3,
            bsdf: DiffuseBrdf,
        }

        impl EmissionShader {
            pub fn new(emission: Float3) -> Self {
                Self {
                    emission: emission,
                    bsdf: DiffuseBrdf,
                }
            }
        }

        impl Shader for EmissionShader {
            fn sample(&self, incoming: &Ray, intersection: &Intersection, radiance: Float3, throughput: Float3) -> ShaderSample {
                let BsdfSample { incoming: outgoing_direction, pdf: _ } = self.bsdf.sample(-incoming.direction, intersection.normal);

                ShaderSample {
                    outgoing: Ray::new(incoming.origin + incoming.direction * intersection.t, outgoing_direction),
                    radiance: radiance + throughput * self.emission,
                    throughput: throughput,
                }
            }
        }

        /// A matte material.
        #[derive(Debug)]
        pub struct MatteShader {
            color: Float3,
            bsdf: DiffuseBrdf,
        }

        impl MatteShader {
            pub fn new(color: Float3) -> Self {
                Self {
                    color: color,
                    bsdf: DiffuseBrdf,
                }
            }
        }

        impl Shader for MatteShader {
            fn sample(&self, incoming: &Ray, intersection: &Intersection, radiance: Float3, throughput: Float3) -> ShaderSample {
                let BsdfSample { incoming: outgoing_direction, pdf: _ } = self.bsdf.sample(-incoming.direction, intersection.normal);

                ShaderSample {
                    outgoing: Ray::new(incoming.origin + incoming.direction * intersection.t, outgoing_direction),
                    radiance: radiance,
                    throughput: self.color * throughput,// * pdf,
                }
            }
        }

        /// A specular material.
        #[derive(Debug)]
        pub struct MirrorShader {
            color: Float3,
            bsdf: MirrorBrdf,
        }

        impl MirrorShader {
            pub fn new(color: Float3) -> Self {
                Self {
                    color: color,
                    bsdf: MirrorBrdf,
                }
            }
        }

        impl Shader for MirrorShader {
            fn sample(&self, incoming: &Ray, intersection: &Intersection, radiance: Float3, throughput: Float3) -> ShaderSample {
                let BsdfSample { incoming: outgoing_direction, pdf: _ } = self.bsdf.sample(-incoming.direction, intersection.normal);

                ShaderSample {
                    outgoing: Ray::new(incoming.origin + incoming.direction * intersection.t, outgoing_direction),
                    radiance: radiance,
                    throughput: self.color * throughput,
                }
            }
        }
    }
}

pub mod shapes {
    //! Intersectable shapes.

    use intersection::{Intersectable, Intersection};
    use math::{Float3, Ray};

    /// A basic sphere.  An origin and a radius.
    #[derive(Debug)]
    pub struct Sphere {
        pub origin: Float3,
        pub radius: f32,
    }

    impl Sphere {
        pub fn new(origin: Float3, radius: f32) -> Self {
            Self {
                origin: origin,
                radius: radius,
            }
        }
    }

    impl Intersectable for Sphere {
        fn intersect<'a>(&'a self, ray: &Ray) -> Option<Intersection> {
            use std::mem;

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
                mem::swap(t0, t1);
            }

            if *t0 > ray.t_bounds.1 || *t1 < ray.t_bounds.0 {
                return None;
            }

            let t = if *t0 >= ray.t_bounds.0 {
                *t0
            } else {
                *t1
            };
            if t > ray.t_bounds.1 {
                return None;
            }

            let normal = (r.origin + r.direction * t).normalized();

            Some(Intersection {
                t: t,
                normal: normal,
                intersectable: self,
            })
        }
    }
}
