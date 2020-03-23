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
// Copyright 2020 Chris Foster

//! `rsrt` is a small, extensible ray tracing framework with some useful implementations.  It provides a basic forward path tracer, a sphere,
//! and some simple materials.
//!
//! To build and render the default scene, use
//! ```
//! $ cargo run --release
//! ```
//! To build and view the documentation, use
//! ```
//! $ cargo doc --open
//! ```

const FILENAME: &'static str = "image.ppm";
const IMAGE_WIDTH: usize = 512;
const IMAGE_HEIGHT: usize = 512;
const SPP: usize = 256;
const THREADS: usize = 4;

use std::io::{self, Write};
use std::sync::{Arc, mpsc, Mutex};
use std::thread;
use std::time::Instant;

use self::camera::{Camera, PerspectiveCamera};
use self::integration::{Integrator, PathTracingIntegrator};
use self::math::{Float3, Ray};
use self::sampling::{RandomSampler, Sampler};
use self::scene::{Object, Scene};
use self::shading::{Color, EmissionShader, MatteShader, MirrorShader, Radiance};
use self::shape::Sphere;

/// Renders a sample scene and outputs a .ppm file.
pub fn main() {
    // Where all of our samples will be collected
    let image_buffer_radiance = Arc::new(Mutex::new(vec![Radiance::from_f32(0.0); IMAGE_WIDTH * IMAGE_HEIGHT]));
    let (sample_count_sender, sample_count_receiver) = mpsc::channel();

    // This object will compute the rendering integral
    let integrator = Arc::new(PathTracingIntegrator);

    // The objects we want to render
    let scene = Arc::new(create_scene());

    // Turns image positions into rays to cast into the scene
    let camera = Arc::new(PerspectiveCamera::new(
        Ray::new(
            Float3::new(0.0, -2.0, 2.5),              // position
            Float3::new(0.0, 4.0, -1.0).normalized(), // look
        ),
        (IMAGE_WIDTH, IMAGE_HEIGHT),
    ));

    let start_time = Instant::now();

    // Start some number of render threads.  Each thread will render a roughly equal number of samples,
    // buffering locally before adding its samples to the final collection at the end.
    let thread_handles = (0..THREADS).map(|i| {
        // Each thread needs its own copy of the main objects
        let image_buffer_radiance = image_buffer_radiance.clone();
        let sample_count_sender = sample_count_sender.clone();
        let integrator = integrator.clone();
        let scene = scene.clone();
        let camera = camera.clone();

        // Each thread will render an equal number of samples, except for the final thread, which
        // will render whatever the remainder is.
        let thread_spp = if i < THREADS - 1 {
            SPP / THREADS
        } else {
            SPP / THREADS + SPP % THREADS
        };
        let thread_scale = thread_spp as f32 / SPP as f32;

        thread::spawn(move || {
            let mut sampler = RandomSampler;

            let thread_buffer = render(&*integrator, &*scene, &*camera, &mut sampler, thread_spp, sample_count_sender);

            // Add the contributions to the global pool
            image_buffer_radiance.lock().unwrap()
                .iter_mut()
                .enumerate()
                .for_each(|(i, pixel)| *pixel = *pixel + thread_buffer[i] * thread_scale);
        })
    }).collect::<Vec<_>>();

    // This thread gets notified each time a render thread has completed a batch of samples.
    // Report progress as it's made.
    let report_thread = thread::spawn(move || {
        const TOTAL_SAMPLES: usize = SPP * IMAGE_WIDTH * IMAGE_HEIGHT;
        const REPORT_INTERVAL: usize = TOTAL_SAMPLES / 10;

        println!("Rendering with {} thread{}...", THREADS, if THREADS > 1 { "s" } else { "" });
        let mut samples = 0;
        while samples < TOTAL_SAMPLES {
            let new_samples = sample_count_receiver.recv().unwrap();

            let next_landmark = (samples / REPORT_INTERVAL + 1) * REPORT_INTERVAL;
            if samples + new_samples >= next_landmark {
                println!("  {:3}%", next_landmark / REPORT_INTERVAL * 10);
            }

            samples += new_samples;
        }

        // Once all samples have been rendered, print some stats
        let seconds = start_time.elapsed().as_secs_f64();
        let (rate, prefix) = {
            let rate = samples as f64 / seconds;
            if rate < 1000.0 {
                (rate, "")
            } else if rate < 1_000_000.0 {
                (rate / 1000.0, "kilo")
            } else {
                (rate / 1_000_000.0, "mega")
            }
        };
        println!("Rendered {} samples in {:.2} seconds, at {:.2} {}samples/second.", samples, seconds, rate, prefix);
    });

    // Wait for everything to finish
    thread_handles.into_iter().for_each(|t| t.join().unwrap());
    report_thread.join().unwrap();

    println!("Writing file...");
    let image_buffer_u8 = clamp_and_quantize(&*image_buffer_radiance.lock().unwrap());
    write_ppm_file(FILENAME, &image_buffer_u8).expect("Could not write file");
    println!("Wrote file \"{}\".", FILENAME);
}

/// Creates a sample [`Scene`] definition, containing some spheres with different material properties.
///
/// [`Scene`]: scene/struct.Scene.html
pub fn create_scene() -> Scene {
    Scene::new(vec![
        Box::new(Object::new(
            Sphere::new(
                Float3::new(-1.0, 3.0, 0.5),    // origin
                0.5,                            // radius
            ), EmissionShader::new(
                Radiance::from_rgb(5.0, 4.0, 3.0), // emission
            ),
        )),
        Box::new(Object::new(
            Sphere::new(
                Float3::new(1.5, 4.0, 1.0),     // origin
                1.0,                            // radius
            ), MatteShader::new(
                Color::from_rgb(0.2, 0.9, 0.2), // color
            ),
        )),
        Box::new(Object::new(
            Sphere::new(
                Float3::new(0.35, 3.4, 0.8),    // origin
                0.2,                            // radius
            ), MirrorShader::new(
                Color::from_rgb(0.7, 0.1, 0.1), // color
            ),
        )),
        Box::new(Object::new(
            Sphere::new(
                Float3::new(0.0, 6.0, 1.5),     // origin
                1.5,                            // radius
            ), MirrorShader::new(
                Color::from_rgb(0.8, 0.8, 0.8), // color
            ),
        )),
        Box::new(Object::new(
            Sphere::new(
                Float3::new(0.0, 5.0, -5000.0), // origin
                5000.0,                         // radius
            ), MatteShader::new(
                Color::from_rgb(0.7, 0.6, 0.6), // color
            ),
        )),
        Box::new(Object::new(
            Sphere::new(
                Float3::new(0.0, 0.0, 0.0),     // origin
                5000.0,                         // radius
            ), EmissionShader::new(
                Radiance::from_rgb(0.825, 0.875, 0.95), // emission
            ),
        )),
    ])
}

/// Takes the [`Radiance`] samples returned from the rendering process and returns a buffer
/// filled with RGB values, 0-255.
///
/// [`Radiance`]: shading/type.Radiance.html
pub fn clamp_and_quantize(image_buffer: &[Radiance]) -> Vec<u8> {
    let mut bytes: Vec<u8> = vec![0; IMAGE_WIDTH * IMAGE_HEIGHT * 3];

    for (index, pixel) in image_buffer.iter().enumerate() {
        bytes[index * 3 + 0] = math::f32_to_255(pixel.x);
        bytes[index * 3 + 1] = math::f32_to_255(pixel.y);
        bytes[index * 3 + 2] = math::f32_to_255(pixel.z);
    }

    bytes
}

/// Collects and returns [`Radiance`] samples from the [`Integrator`], over the [`Scene'].
///
/// This function renders the image to the samples/pixel specified in `spp`, periodically reporting its progress
/// into `sample_count_sender`.
///
/// [`Radiance`]: shading/type.Radiance.html
/// [`Integrator`]: integration/trait.Integrator.html
/// [`Scene`]: scene/struct.Scene.html
pub fn render<I, C, S>(
    integrator: &I,
    scene: &Scene,
    camera: &C,
    sampler: &mut S,
    spp: usize,
    sample_count_sender: mpsc::Sender<usize>,
) -> Vec<Radiance> where
    I: Integrator,
    C: Camera,
    S: Sampler,
{
    let mut radiance_buffer = vec![Radiance::from_f32(0.0); IMAGE_WIDTH * IMAGE_HEIGHT];
    let sample_scale = 1.0 / spp as f32;
    for y in 0..IMAGE_HEIGHT {
        for x in 0..IMAGE_WIDTH {
            for _ in 0..spp {
                let ray = camera.generate_ray((x, y), sampler);
                let radiance = integrator.integrate(scene, ray, sampler);

                let pixel = &mut radiance_buffer[y * IMAGE_WIDTH + x];
                *pixel = *pixel + radiance * sample_scale;

                sampler.next_sample();
            }
            sampler.reset();
        }
        sample_count_sender.send(IMAGE_WIDTH * spp).unwrap();
    }

    radiance_buffer
}

/// Writes a quantized image buffer into a .ppm file.
pub fn write_ppm_file(filename: &str, image_buffer: &[u8]) -> io::Result<()> {
    let image_file_path = std::path::Path::new(filename);
    let mut image_file = std::fs::File::create(&image_file_path)?;

    let image_file_header = format!("P6\n{} {}\n{}\n", IMAGE_WIDTH, IMAGE_HEIGHT, 255);
    image_file.write_all(image_file_header.as_bytes())?;
    image_file.write_all(image_buffer)
}

pub mod camera {
    //! Cameras are used to generate rays from image positions.

    use crate::math::{Float3, Ray};
    use crate::sampling::Sampler;

    pub trait Camera {
        /// Generates a [`Ray`] from the image coordinate `pixel`.  The implementation may use values returned by the
        /// [`Sampler`] to adjust the generated ray.
        ///
        /// [`Ray`]: ../math/struct.Ray.html
        /// [`Sampler`]: ../sampling/trait.Sampler.html
        fn generate_ray(&self, pixel: (usize, usize), sampler: &mut dyn Sampler) -> Ray;
    }

    /// Implements a point camera with a FOV of 45 degrees.
    pub struct PerspectiveCamera {
        look: Ray,
        image_size: (usize, usize),

        view_to_world_x: Float3,
        view_to_world_y: Float3,
        view_to_world_z: Float3,
    }

    impl PerspectiveCamera {
        /// Creates a camera centered at `look`'s origin, and oriented `look`'s direction.
        /// `image_size` describes the size of the grid through which to project samples.
        pub fn new(look: Ray, image_size: (usize, usize)) -> Self {
            let view_to_world_z = look.direction;
            let view_to_world_x = view_to_world_z.cross(&Float3::new(0.0, 0.0, 1.0)).normalized();
            let view_to_world_y = view_to_world_x.cross(&view_to_world_z);

            Self {
                look,
                image_size,
                view_to_world_x,
                view_to_world_y,
                view_to_world_z,
            }
        }
    }

    impl Camera for PerspectiveCamera {
        fn generate_ray(&self, pixel: (usize, usize), sampler: &mut dyn Sampler) -> Ray {
            let (x, y) = (sampler.next_dimension(), sampler.next_dimension());

            let image_x = (pixel.0 as f32 + x) / self.image_size.0 as f32;
            let image_y = (pixel.1 as f32 + y) / self.image_size.1 as f32;

            let view_ray_direction = Float3::new(
                2.0 * image_x - 1.0,
                1.0 - 2.0 * image_y,
                2.0,
            ).normalized();

            Ray::new(
                self.look.origin,
                self.view_to_world_x * view_ray_direction.x +
                self.view_to_world_y * view_ray_direction.y +
                self.view_to_world_z * view_ray_direction.z,
            )
        }
    }
}

pub mod integration {
    //! Trait and implementations that attempt to evaluate the integral of the rendering equation.

    use crate::math::Ray;
    use crate::sampling::Sampler;
    use crate::scene::Scene;
    use crate::shading::{Radiance, Shadeable, ShaderSample};

    pub trait Integrator {
        /// Integrate along the given ray, returning an estimate of the incoming radiance.
        fn integrate(&self, scene: &Scene, ray: Ray, sampler: &mut dyn Sampler) -> Radiance;
    }

    /// A simple forward path tracer.  Rays bounce a minimum of 3 times, and then are subjected to Russian Roulette termination.
    pub struct PathTracingIntegrator;

    impl Integrator for PathTracingIntegrator {
        fn integrate(&self, scene: &Scene, mut ray: Ray, sampler: &mut dyn Sampler) -> Radiance {
            let mut radiance = Radiance::from_f32(0.0);
            let mut throughput = Radiance::from_f32(1.0);

            let mut bounces = 0;
            'cast_ray: while let Some(mut point) = scene.cast_ray(&ray) {
                // If we've hit the backside of a surface, flip the normal for the sake of our calculations
                if ray.direction.dot(&point.intersection.normal) > 0.0 {
                    point.intersection.normal = point.intersection.normal * -1.0;
                }

                // Sample a random bounce off the surface
                let ShaderSample {
                    outgoing: next_ray,
                    radiance: next_radiance,
                    throughput: next_throughput,
                } = point.shader.sample(
                    &ray,
                    &point.intersection,
                    radiance,
                    throughput,
                    (sampler.next_dimension(), sampler.next_dimension()),
                );

                radiance = next_radiance;
                throughput = next_throughput;
                bounces += 1;

                let continuation_probability = throughput.average();
                if continuation_probability == 0.0 {
                    break 'cast_ray;
                } else if bounces > 3 { // If we've bounced more than three times, terminate rays with random chance
                    if continuation_probability < sampler.next_dimension() {
                        break 'cast_ray;
                    }
                    throughput = throughput * continuation_probability;
                }

                // Nudge the next ray slightly off the surface to prevent autocollisions
                ray = Ray::new(next_ray.origin + point.intersection.normal * 0.001, next_ray.direction);
            }

            // The accumulated radiance is what we're after
            radiance
        }
    }
}

pub mod math {
    //! A couple geometry-related structures and common operations.

    use std::ops::{Add, Sub, Mul, Neg};

    #[derive(Clone, Copy, Debug)]
    pub struct Float3 {
        pub x: f32,
        pub y: f32,
        pub z: f32,
    }

    /// A vector of 3 floating point numbers.
    impl Float3 {
        pub fn new(x: f32, y: f32, z: f32) -> Self {
            Self { x, y, z }
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
    }

    impl Ray {
        pub fn new(origin: Float3, direction: Float3) -> Self {
            Self {
                origin: origin,
                direction: direction,
            }
        }
    }

    /// A surface that can be detected by a ray.
    pub trait Intersectable {
        fn intersect(&self, ray: &Ray) -> Option<Intersection>;
    }

    /// An intersection of a surface by a ray.
    pub struct Intersection {
        /// The time along the ray at which the intersection occurred.
        pub t: f32,
        /// The normal of the surface at the intersection point.
        pub normal: Float3,
    }

    /// Clamp an f32 value to [0, 1] and quantize to 8 bits.
    pub fn f32_to_255(value: f32) -> u8 {
        (value.max(0.0).min(1.0) * 255.0 + 0.5) as u8
    }
}

pub mod sampling {
    //! Trait and implementations for generating pseudo-random numbers.

    use rand::Rng;

    /// A sample generator.  Each sample is a vector of values, distributed across the multidimensional
    /// sample space in a manner dependent on the implementation.
    pub trait Sampler {
        /// Returns the next dimension of the current sample.
        fn next_dimension(&mut self) -> f32;
        /// Advance to the next sample.
        fn next_sample(&mut self);
        fn reset(&mut self);
    }

    /// Returns uniform random values for each dimension of each sample.
    pub struct RandomSampler;

    impl Sampler for RandomSampler {
        fn next_dimension(&mut self) -> f32 {
            rand::thread_rng().gen()
        }

        fn next_sample(&mut self) { }

        fn reset(&mut self) { }
    }
}

pub mod scattering {
    //! Trait and implementations that define the scattering behavior of light across different surfaces.

    use std::f32::consts::{FRAC_1_PI, PI};

    use crate::math::Float3;
    use crate::shading::Radiance;

    /// A Bidirectional Scattering Distribution Function defines a surface's light-scattering properties.
    pub trait Bsdf {
        /// Returns a randomly sampled incoming direction for light scattered in the outgoing direction.
        fn sample_incoming(&self, outgoing: Float3, normal: Float3, sampled_values: (f32, f32)) -> BsdfSample;
        /// Returns the radiance transmitted from `incoming` to `outgoing`.
        fn f(&self, outgoing: Float3, normal: Float3, incoming: Float3) -> Radiance;
        /// Returns the pdf of the light path from `incoming` to `outgoing`.
        fn pdf(&self, outgoing: Float3, normal: Float3, incoming: Float3) -> f32;
    }

    pub struct BsdfSample {
        pub direction: Float3,
        pub f: Radiance,
        pub pdf: f32,
    }

    /// BRDF for perfect diffuse reflectance.
    pub struct DiffuseBrdf;

    impl Bsdf for DiffuseBrdf {
        fn sample_incoming(&self, _outgoing: Float3, normal: Float3, sampled_values: (f32, f32)) -> BsdfSample {
            let r = sampled_values.0.sqrt();
            let theta = 2.0 * PI * sampled_values.1;

            let x = r * theta.cos();
            let y = r * theta.sin();
            let z = (0.0f32).max(1.0 - x * x - y * y).sqrt();

            let (tangent, bitangent) = normal.make_orthonormals();
            let incoming = tangent * x + bitangent * y + normal * z;

            let pdf = z * FRAC_1_PI;

            BsdfSample {
                direction: incoming,
                f: Radiance::from_f32(pdf),
                pdf: pdf,
            }
        }

        fn f(&self, outgoing: Float3, normal: Float3, incoming: Float3) -> Radiance {
            Radiance::from_f32(self.pdf(outgoing, normal, incoming))
        }

        fn pdf(&self, _outgoing: Float3, normal: Float3, incoming: Float3) -> f32 {
            (0.0f32).max(normal.dot(&incoming)) * FRAC_1_PI
        }
    }

    /// BRDF for perfect specular reflectance.
    pub struct MirrorBrdf;

    impl Bsdf for MirrorBrdf {
        fn sample_incoming(&self, outgoing: Float3, normal: Float3, _sampled_values: (f32, f32)) -> BsdfSample {
            BsdfSample {
                direction: normal * 2.0 * normal.dot(&outgoing) - outgoing,
                f: Radiance::from_f32(1.0),
                pdf: 1.0,
            }
        }

        fn f(&self, _outgoing: Float3, _normal: Float3, _incoming: Float3) -> Radiance {
            Radiance::from_f32(0.0)
        }

        fn pdf(&self, _outgoing: Float3, _normal: Float3, _incoming: Float3) -> f32 {
            0.0
        }
    }
}

pub mod scene {
    //! Scene and object definitions.

    use crate::math::{Intersectable, Ray};
    use crate::shading::{Shadeable, Shader, ShadingPoint};

    /// The combination of a shader with an intersectable surface.
    pub struct Object<I, S> where
        I: Intersectable,
        S: Shader,
    {
        intersectable: I,
        shader: S,
    }

    impl<I, S> Object<I, S> where
        I: Intersectable,
        S: Shader,
    {
        pub fn new(intersectable: I, shader: S) -> Self {
            Self {
                intersectable,
                shader,
            }
        }
    }

    impl<I, S> Shadeable for Object<I, S> where
        I: Intersectable,
        S: Shader,
    {
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
    pub struct Scene {
        pub objects: Vec<Box<dyn Shadeable + Send + Sync>>,
    }

    impl Scene {
        pub fn new(objects: Vec<Box<dyn Shadeable + Send + Sync>>) -> Self {
            Self {
                objects,
            }
        }
    }

    impl Shadeable for Scene {
        fn cast_ray(&self, ray: &Ray) -> Option<ShadingPoint> {
            let mut closest_point: Option<ShadingPoint> = None;

            for object in &self.objects {
                if let Some(point) = object.cast_ray(ray) {
                    if closest_point.is_none() || point.intersection.t < closest_point.as_ref().unwrap().intersection.t {
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

    use crate::math::{Float3, Intersection, Ray};
    use crate::scattering::{Bsdf, BsdfSample, DiffuseBrdf, MirrorBrdf};

    pub type Radiance = Float3;

    impl Radiance {
        pub fn from_f32(value: f32) -> Self {
            Self::new(value, value, value)
        }
    }

    pub type Color = Radiance;

    impl Color {
        pub fn from_rgb(r: f32, g: f32, b: f32) -> Self {
            Self::new(r, g, b)
        }
    }

    /// A material.
    pub trait Shader {
        /// Returns a new ray after bouncing the old one off of the material, along with
        /// the new radiance and throughput.
        fn sample(&self, incoming: &Ray, intersection: &Intersection, radiance: Radiance, throughput: Radiance, sampled_values: (f32, f32)) -> ShaderSample;
    }

    pub struct ShaderSample {
        pub outgoing: Ray,
        pub radiance: Radiance,
        pub throughput: Radiance,
    }

    /// An object that can be detected by a ray and has material properties.
    pub trait Shadeable {
        fn cast_ray(&self, ray: &Ray) -> Option<ShadingPoint>;
    }

    pub struct ShadingPoint<'a> {
        pub intersection: Intersection,
        pub shader: &'a dyn Shader,
    }

    /// A material that emits light.
    pub struct EmissionShader {
        emission: Radiance,
    }

    impl EmissionShader {
        pub fn new(emission: Radiance) -> Self {
            Self {
                emission,
            }
        }
    }

    impl Shader for EmissionShader {
        fn sample(&self, _incoming: &Ray, _: &Intersection, radiance: Radiance, throughput: Radiance, _sampled_values: (f32, f32)) -> ShaderSample {
            ShaderSample {
                outgoing: Ray::new(
                    Float3::new(0.0, 0.0, 0.0),
                    Float3::new(0.0, 0.0, 0.0),
                ), // Reflection is undefined for emission shaders, so this will never be used
                radiance: radiance + throughput * self.emission,
                throughput: Radiance::from_f32(0.0), // No throughput for emission shaders
            }
        }
    }

    /// A matte material.
    pub struct MatteShader {
        color: Color,
        bsdf: DiffuseBrdf,
    }

    impl MatteShader {
        pub fn new(color: Color) -> Self {
            Self {
                color,
                bsdf: DiffuseBrdf,
            }
        }
    }

    impl Shader for MatteShader {
        fn sample(&self, incoming: &Ray, intersection: &Intersection, radiance: Float3, throughput: Float3, sampled_values: (f32, f32)) -> ShaderSample {
            let BsdfSample {
                direction: outgoing_direction,
                f: _,
                pdf: _,
            } = self.bsdf.sample_incoming(-incoming.direction, intersection.normal, sampled_values);

            ShaderSample {
                outgoing: Ray::new(incoming.origin + incoming.direction * intersection.t, outgoing_direction),
                radiance,
                throughput: self.color * throughput, // * f * (1.0 / pdf)
            }
        }
    }

    /// A specular material.
    pub struct MirrorShader {
        color: Color,
        bsdf: MirrorBrdf,
    }

    impl MirrorShader {
        pub fn new(color: Color) -> Self {
            Self {
                color,
                bsdf: MirrorBrdf,
            }
        }
    }

    impl Shader for MirrorShader {
        fn sample(&self, incoming: &Ray, intersection: &Intersection, radiance: Float3, throughput: Float3, sampled_values: (f32, f32)) -> ShaderSample {
            let BsdfSample {
                direction: outgoing_direction,
                f: _,
                pdf: _,
            } = self.bsdf.sample_incoming(-incoming.direction, intersection.normal, sampled_values);

            ShaderSample {
                outgoing: Ray::new(incoming.origin + incoming.direction * intersection.t, outgoing_direction),
                radiance,
                throughput: self.color * throughput, // * f * (1.0 / pdf)
            }
        }
    }
}

pub mod shape {
    //! Intersectable shapes.

    use crate::math::{Float3, Intersectable, Intersection, Ray};

    /// A sphere, defined by an origin and a radius.
    pub struct Sphere {
        pub origin: Float3,
        pub radius: f32,
    }

    impl Sphere {
        pub fn new(origin: Float3, radius: f32) -> Self {
            Self {
                origin,
                radius,
            }
        }
    }

    impl Intersectable for Sphere {
        fn intersect(&self, ray: &Ray) -> Option<Intersection> {
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

            if *t1 < 0.0 {
                return None;
            }

            let t = if *t0 >= 0.0 {
                *t0
            } else {
                *t1
            };

            let normal = (r.origin + r.direction * t).normalized();

            Some(Intersection {
                t: t,
                normal: normal,
            })
        }
    }
}