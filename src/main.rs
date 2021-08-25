#![feature(option_zip)]

use std::path::Path;
use glam::{Vec3, Vec4};
use rayon::prelude::*;

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
       let reflected_dir = self.direction - *hit_normal * 2.0 * hit_normal.dot(self.direction); 
       Ray::new(*hit_pos, reflected_dir)
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

  /// Create a new color from a vector
  fn from_vec4(v: Vec4) -> Color {
    Color::new(v.x, v.y, v.z, v.w)
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
    fn intersect_ray(&self, ray: &Ray) -> Option<Hit>;
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
    fn intersect_ray(&self, ray: &Ray) -> Option<Hit> {
        let mut closest_hit: Option<Hit> = None;

        for object in self.objects.iter() {
            let new_hit = object.intersect_ray(ray);
            closest_hit = closest_hit.or(new_hit).zip_with(new_hit, Hit::closest);
        }

        closest_hit
    }
}

/// Sphere structure
#[derive(Copy, Clone)]
pub struct Sphere {
    origin: Vec3,
    radius: f32
}

impl Sphere {
    /// Create a new sphere
    fn new(origin: Vec3, radius: f32) -> Sphere {
        Sphere { origin, radius }
    }
}

impl Traceable for Sphere {
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
    fn intersect_ray(&self, ray: &Ray) -> Option<Hit> {
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
            // could be 1 or 2 roots, but we only care about the first one
            let t = (-b - discriminant.sqrt()) / (2.0 * a);
            let pos = ray.at(t);
            let nrm = (pos - self.origin) / self.radius;
            let front_face = ray.direction.dot(nrm) < 0.0;

            Some(Hit::new(t, pos, nrm, front_face))
        }
    }
}

/// Entrypoint
fn main() {
    // Configuration
    const OUTPUT_WIDTH: usize = 800;
    const OUTPUT_HEIGHT: usize = 600;
    const OUTPUT_FILENAME: &str = &"image.png";

    // Calculated values
    const ASPECT_RATIO: f32 = (OUTPUT_WIDTH as f32) / (OUTPUT_HEIGHT as f32);

    // Create buffer
    let mut buf = vec![Color::new(0.0, 0.0, 0.0, 1.0); OUTPUT_WIDTH * OUTPUT_HEIGHT];

    // Viewport parameters
    let viewport_height = 2.0;
    let viewport_width = viewport_height * ASPECT_RATIO;
    let focal_length = 1.0;

    let origin = Vec3::new(0.0, 0.0, 0.0);
    let lower_left_corner = origin
        - Vec3::new(viewport_width, viewport_height, 0.0) / 2.0
        - Vec3::new(0.0, 0.0, focal_length);

    // World definition
    let world = TraceableList::new(vec![
        Box::new(Sphere::new(Vec3::new(0.0, 0.0, -5.0), 2.0))
    ]);

    // Draw image
    buf.par_iter_mut().enumerate().for_each(|(i, color)| {
        let x = i % OUTPUT_WIDTH;
        let y = i / OUTPUT_WIDTH;

        let u = (x as f32) / (OUTPUT_WIDTH as f32);
        let v = (y as f32) / (OUTPUT_HEIGHT as f32);

        let ray_dir = lower_left_corner + Vec3::new(u * viewport_width, v * viewport_height, 0.0) - origin;
        let ray = Ray::new(origin, ray_dir);

        let out_col = trace(&ray, &world, 0);
        *color = Color::new(out_col.x, out_col.y, out_col.z, 1.0);
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
    return top_color.lerp(bottom_color, ray.direction.y);
}

/// Trace the scene
fn trace(ray: &Ray, scene: &dyn Traceable, depth: i32) -> Vec3 {
    if depth > 7 {
        return Vec3::new(0.0, 0.0, 0.0);
    }

    let hit = scene.intersect_ray(ray);

    match hit {
        Some(hit) => {
            let diffuse = hit.normal.dot(Vec3::new(0.0, 0.0, 1.0));
            Vec3::new(diffuse, diffuse, diffuse)
        },
        None => {
            let col = background(ray);
            Vec3::new(col.x, col.y, col.z)
        }
    }
}
