#![feature(option_zip)]

use std::path::Path;
use glam::{Vec3, Vec4};
use rayon::prelude::*;
use rand::Rng;

/// Module for random generation functions
mod random_gen {
    use glam::Vec3;
    use rand::Rng;

    /// Get a point in a unit sphere using rejection sampling
    pub fn random_point_in_unit_sphere() -> Vec3 {
        let mut rng = rand::thread_rng();

        loop {
            let point = Vec3::new(
                rng.gen_range(-1.0.. 1.0),
                rng.gen_range(-1.0.. 1.0),
                rng.gen_range(-1.0.. 1.0));

            if point.length_squared() <= 1.0 {
                return point;
            }
        }
    }

    /// Get a random vector in a hemisphere
    pub fn random_point_in_unit_hemisphere(normal: &Vec3) -> Vec3 {
        let in_unit_sphere = random_point_in_unit_sphere();
        if in_unit_sphere.dot(*normal) > 0.0 {
            in_unit_sphere
        } else {
            -in_unit_sphere
        }
    }

    /// Get a random unit vector
    pub fn random_unit_vector() -> Vec3 {
        random_point_in_unit_sphere().normalize()
    }
}

/// Ray structure
#[derive(Copy, Clone)]
pub struct Ray {
    origin: Vec3,
    direction: Vec3
}

impl Ray {
    /// Create a new ray
    fn new(origin: Vec3, direction: Vec3) -> Ray {
        Ray { origin, direction }
    }

    /// Get the position a distance along the ray
    fn at(&self, t: f32) -> Vec3 {
        return self.origin + t * self.direction;
    }

    /// Reflect the ray
    /// Reflected ray direction = I - 2.0 * dot(N, I) * N
    fn reflect(&self, hit_pos: &Vec3, hit_normal: &Vec3) -> Ray {
        let direction_normalised = self.direction.normalize();
        let reflected_pos = *hit_pos;
        let reflected_dir = direction_normalised - *hit_normal * 2.0 * hit_normal.dot(direction_normalised); 
        Ray::new(reflected_pos, reflected_dir)
    }
}

/// Camera
#[derive(Copy, Clone)]
pub struct Camera {
    origin: Vec3,
    viewport_width: f32,
    viewport_height: f32,
    focal_length: f32
}

impl Camera {
    /// Create a new camera
    fn new(origin: Vec3, aspect_ratio: f32) -> Camera {
        let viewport_height = 2.0;
        let viewport_width = viewport_height * aspect_ratio;
        let focal_length = 1.0;

        Camera { origin, viewport_width, viewport_height, focal_length }
    }

    /// Create a new ray
    fn create_ray(&self, u: f32, v: f32) -> Ray {
        let lower_left_corner = self.origin
            - Vec3::new(self.viewport_width, self.viewport_height, 0.0) / 2.0
            - Vec3::new(0.0, 0.0, self.focal_length);

        let ray_dir = lower_left_corner
            + Vec3::new(u * self.viewport_width, v * self.viewport_height, 0.0)
            - self.origin;

        Ray::new(self.origin, ray_dir)
    }
}

/// Colors
#[derive(Copy, Clone)]
pub struct Color {
    r: f32,
    g: f32,
    b: f32,
    a: f32
}

impl Color {
    /// Create a new color
    fn new(r: f32, g: f32, b: f32, a: f32) -> Color {
        Color { r, g, b, a }
    }

    /// Create a new color from a Vec3
    fn from_vec3(v: Vec3) -> Color {
        Color::new(v.x, v.y, v.z, 1.0)
    }

    /// Create a new color from a Vec4
    fn from_vec4(v: Vec4) -> Color {
        Color::new(v.x, v.y, v.z, v.w)
    }

    /// Convert a color to a Vec4
    fn to_vec4(&self) -> Vec4 {
        Vec4::new(self.r, self.g, self.b, self.a)
    }

    /// Convert a normalised f32 color component to a 0..255 u8
    fn comp_to_u8_color(value: &f32) -> u8 {
        match value {
            v if v <= &0.0 => 0,
            v if v >= &1.0 => 255,
            v              => (v * 256.0) as u8
        }
    }

    /// Convert to Vec<u8>
    fn to_u8_vec(&self) -> Vec<u8> {
        vec!(
            Color::comp_to_u8_color(&self.r),
            Color::comp_to_u8_color(&self.g),
            Color::comp_to_u8_color(&self.b),
            Color::comp_to_u8_color(&self.a)
        )
    }

    /// Convert a Vec<Color> to [u8] (with 4 times as many elements)
    fn vec_to_u8_slice(vec: &Vec<Color>) -> Vec<u8> {
        vec.iter().flat_map(Color::to_u8_vec).collect()
    }
}

/// A hit
#[derive(Copy, Clone)]
pub struct Hit {
    t: f32,
    position: Vec3,
    normal: Vec3,
    front_face: bool
}

impl Hit {
    /// Create a new hit
    pub fn new(t: f32, position: Vec3, normal: Vec3, front_face: bool) -> Hit {
        Hit { t, position, normal, front_face }
    }

    /// Get the closest of two hits
    pub fn closest(self, other: Hit) -> Hit {
        if self.t < other.t {
            self
        } else {
            other
        }
    }
}

/// Traceable trait
pub trait Traceable {
    fn intersect_ray(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<(Hit, Box<&dyn Material>)>;
}

/// Traceable list structure
pub struct TraceableList {
    objects: Vec<Box<dyn Traceable + Sync + Send>>
}

impl TraceableList {
    // Create new TraceableList
    fn new(objects: Vec<Box<dyn Traceable + Sync + Send>>) -> TraceableList {
        TraceableList { objects }
    }
}

impl Traceable for TraceableList {
    // Intersect a ray with the traceables in the list
    fn intersect_ray(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<(Hit, Box<&dyn Material>)> {
        let mut closest_t: f32 = f32::MAX;
        let mut closest_hit: Option<(Hit, Box<&dyn Material>)> = None;

        for object in self.objects.iter() {
            let new_hit = object.intersect_ray(ray, t_min, t_max);

            match new_hit {
                Some((hit, _)) if hit.t < closest_t => {
                    closest_t = hit.t;
                    closest_hit = new_hit;
                },
                _ => {}
            }
        }

        closest_hit
    }
}

/// Sphere structure
#[derive(Copy, Clone)]
pub struct Sphere<T: Material> {
    origin: Vec3,
    radius: f32,
    material: T
}

impl<T: Material> Sphere<T> {
    /// Create a new sphere
    fn new(origin: Vec3, radius: f32, material: T) -> Sphere<T> {
        Sphere { origin, radius, material }
    }
}

impl<T: Material> Traceable for Sphere<T> {
    /// Intersect a ray with the sphere
    /// Points on a sphere:
    ///     (x - Cx)^2 + (y - Cy)^2 + (z - Cz)^2 = r^2
    ///     = (P - C)^2 = r^2
    /// Points along ray:
    ///     P(t) = o + td
    /// Combined:
    ///     (P(t) - C) . (P(t) - C) = r^2
    ///     (o + td - C) . (o + td - C) = r^2
    ///     (td + (o - C)) . (td + o - C) = r^2
    ///     (d.d)t^2 + 2(o - C).d * t + (o - C).(o - C) - r^2 = 0
    /// Solve quadratic equation, # roots = number of hits
    fn intersect_ray(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<(Hit, Box<&dyn Material>)> {
        let o_minus_c = ray.origin - self.origin;

        // at^2 + bt + c = 0
        let a = Vec3::dot(ray.direction, ray.direction);
        let b = 2.0 * Vec3::dot(o_minus_c, ray.direction);
        let c = Vec3::dot(o_minus_c, o_minus_c) - self.radius * self.radius;

        // sign of the discriminant tells us the number of roots
        // zero = 1 root, positive = 2 roots
        let discriminant = b * b - 4.0 * a * c;

        if discriminant < 0.0 {
            // negative = no roots
            None
        } else {
            // could be 1 or 2 roots, find the nearest one that's within our range
            let discriminant_sqrt = discriminant.sqrt();

            // start with the negative root
            let mut nearest_root = (-b - discriminant_sqrt) / (2.0 * a);

            // If it's behind t_min or beyond t_max, try the second root
            if nearest_root < t_min || nearest_root > t_max {
                nearest_root = (-b + discriminant_sqrt) / (2.0 * a);
            }

            if nearest_root < t_min || nearest_root > t_max {
                None
            } else {
                let pos = ray.at(nearest_root);
                let outward_normal = (pos - self.origin) / self.radius;
                let front_face = ray.direction.dot(outward_normal) < 0.0;
                let nrm = if front_face { outward_normal } else { -outward_normal };

                let hit = Hit::new(nearest_root, pos, nrm, front_face);

                Some((hit, Box::new(&self.material)))
            }
        }
    }
}

/// Material structure
pub trait Material {
    /// Scatter a ray
    fn scatter(&self, ray_in: &Ray, hit: &Hit) -> Option<(Ray, Vec3)>;
}

/// A lambertian material
#[derive(Copy, Clone)]
pub struct Lambertian {
    albedo: Vec3
}

impl Lambertian {
    /// Create a new instance of a lambertian material
    fn new(albedo: Vec3) -> Lambertian {
        Lambertian { albedo }
    }
}

impl Material for Lambertian {
    fn scatter(&self, _: &Ray, hit: &Hit) -> Option<(Ray, Vec3)> {
        let scatter_direction = hit.normal + random_gen::random_unit_vector();
        let ray = Ray::new(hit.position, scatter_direction);

        Some((ray, self.albedo))
    }
}

/// A chrome material
#[derive(Copy, Clone)]
pub struct Chrome {
    albedo: Vec3
}

impl Chrome {
    /// Create a new instance of a chrome material
    fn new(albedo: Vec3) -> Chrome {
        Chrome { albedo }
    }
}

impl Material for Chrome {
    fn scatter(&self, ray_in: &Ray, hit: &Hit) -> Option<(Ray, Vec3)> {
        let ray = ray_in.reflect(&hit.position, &hit.normal);
        Some((ray, self.albedo))
    }
}

/// Entrypoint
fn main() {
    // Configuration
    const OUTPUT_WIDTH: usize = 800;
    const OUTPUT_HEIGHT: usize = 600;
    const OUTPUT_FILENAME: &str = &"image.png";
    const SAMPLES_PER_PIXEL : usize = 100;

    // Calculated values
    const ASPECT_RATIO: f32 = (OUTPUT_WIDTH as f32) / (OUTPUT_HEIGHT as f32);

    // Create buffer
    let mut buf = vec![Color::new(0.0, 0.0, 0.0, 1.0); OUTPUT_WIDTH * OUTPUT_HEIGHT];

    // Create camera
    let camera = Camera::new(Vec3::new(0.0, 0.0, 0.0), ASPECT_RATIO);

    // Materials
    let ground = Lambertian::new(Vec3::new(1.0, 1.0, 0.0));
    let left = Chrome::new(Vec3::new(0.8, 0.8, 0.8));
    let right = Lambertian::new(Vec3::new(0.5, 0.0, 0.0));

    // World definition
    let scene = TraceableList::new(vec![
        Box::new(Sphere::new(Vec3::new(0.0, -100.5, -1.5), 100.0, ground)),
        Box::new(Sphere::new(Vec3::new(-0.5, 0.0, -1.5), 0.5, left)),
        Box::new(Sphere::new(Vec3::new(0.5, 0.0, -1.5), 0.5, right)),
    ]);

    // Draw image
    buf.par_iter_mut().enumerate().for_each(|(i, color)| {
        let x = i % OUTPUT_WIDTH;
        let y = i / OUTPUT_WIDTH;

        let mut rng = rand::thread_rng();

        let mut accumulated = Vec3::new(0.0, 0.0, 0.0);

        for _ in 0..SAMPLES_PER_PIXEL {
            let u_jitter: f32 = rng.gen();
            let v_jitter: f32 = rng.gen();

            let u = (x as f32 + u_jitter) / (OUTPUT_WIDTH as f32);
            let v = 1.0 - (y as f32 + v_jitter) / (OUTPUT_HEIGHT as f32);

            let ray = camera.create_ray(u, v);

            accumulated += trace(&ray, &scene, 0);
        }

        // Divide by number of samples and gamma correct with 'gamma 2' (raised to the power of 1/2)
        let scale = 1.0 / SAMPLES_PER_PIXEL as f32;
        let r = f32::sqrt(scale * accumulated.x);
        let g = f32::sqrt(scale * accumulated.y);
        let b = f32::sqrt(scale * accumulated.z);

        *color = Color::new(r, g, b, 1.0);
    });

    let out_buf: &[u8] = &Color::vec_to_u8_slice(&buf);

    // Save image
    image::save_buffer(&Path::new(OUTPUT_FILENAME), &out_buf,
    OUTPUT_WIDTH as u32, OUTPUT_HEIGHT as u32, image::ColorType::Rgba8)
        .unwrap();
}

/// Shade the background
fn background(ray: &Ray) -> Vec3 {
    let top_color: Vec3 = Vec3::new(0.258, 0.457, 0.727);
    let bottom_color: Vec3 = Vec3::new(0.789, 0.876, 0.922);

    bottom_color.lerp(top_color, 0.5 * (ray.direction.y + 1.0))
}

/// Trace the scene
fn trace<T: Traceable>(ray: &Ray, scene: &T, depth: i32) -> Vec3 {
    if depth > 7 {
        return Vec3::new(0.0, 0.0, 0.0);
    }

    match scene.intersect_ray(ray, 0.0001, f32::MAX) {
        Some((hit, mat)) => {
            // Scatter ray
            match mat.scatter(&ray, &hit) {
                Some((ray, attenuation)) => trace(&ray, scene, depth + 1) * attenuation,
                _                        => Vec3::new(0.0, 0.0, 0.0)
            }
        },
        None => {
            background(ray)
        }
    }
}

