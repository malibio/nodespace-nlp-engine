//! Image processing and multimodal capabilities
//!
//! This module provides image embedding generation using CLIP models via fastembed,
//! EXIF metadata extraction, and image preprocessing for ONNX Runtime.

#[cfg(feature = "multimodal")]
use crate::error::NLPError;
#[cfg(feature = "multimodal")]
use crate::models::DeviceType;
#[cfg(feature = "multimodal")]
use crate::utils::metrics::Timer;
#[cfg(feature = "multimodal")]
use crate::{CameraInfo, ImageMetadata};

#[cfg(feature = "multimodal")]
use chrono::{DateTime, Utc};
#[cfg(feature = "multimodal")]
use dashmap::DashMap;
#[cfg(feature = "multimodal")]
use std::sync::Arc;
#[cfg(feature = "multimodal")]
use std::time::Instant;

#[cfg(feature = "multimodal")]
use exif;

// Type aliases to reduce complexity warnings
#[cfg(feature = "multimodal")]
type ImageEmbeddingCache = Arc<DashMap<String, (Vec<f32>, DateTime<Utc>)>>;

#[cfg(feature = "multimodal")]
type ExifResult = Result<
    (
        Option<DateTime<Utc>>,
        Option<(f64, f64)>,
        Option<CameraInfo>,
        Option<u8>,
    ),
    NLPError,
>;

/// Image embedding generator using CLIP via fastembed
#[cfg(feature = "multimodal")]
pub struct ImageEmbeddingGenerator {
    model: Option<fastembed::ImageEmbedding>,
    #[allow(dead_code)] // Reserved for future use with device selection
    device_type: DeviceType,
    cache: ImageEmbeddingCache,
    cache_ttl_seconds: u64,
    dimensions: usize,
}

#[cfg(feature = "multimodal")]
impl ImageEmbeddingGenerator {
    /// Create new image embedding generator
    pub fn new(device_type: DeviceType) -> Result<Self, NLPError> {
        Ok(Self {
            model: None,
            device_type,
            cache: Arc::new(DashMap::new()),
            cache_ttl_seconds: 3600, // 1 hour default
            dimensions: 512,         // CLIP ViT-B-32 dimensions
        })
    }

    /// Initialize the image embedding model
    pub async fn initialize(&mut self) -> Result<(), NLPError> {
        let _timer = Timer::new("image_embedding_model_initialization");

        tracing::info!("Initializing CLIP image embedding model...");

        // Initialize fastembed ImageEmbedding with CLIP ViT-B-32
        let model = fastembed::ImageEmbedding::try_new(Default::default()).map_err(|e| {
            NLPError::ModelLoading {
                message: format!("Failed to load CLIP model: {}", e),
            }
        })?;

        self.model = Some(model);
        tracing::info!("CLIP image embedding model initialized successfully");
        Ok(())
    }

    /// Generate embedding for image data
    pub async fn generate_embedding(&self, image_data: &[u8]) -> Result<Vec<f32>, NLPError> {
        let _timer = Timer::new("image_embedding_generation");
        let start_time = Instant::now();

        // Check cache first
        let cache_key = self.compute_cache_key(image_data);
        if let Some(cached_entry) = self.cache.get(&cache_key) {
            let (embedding, timestamp) = cached_entry.value();
            let age = Utc::now().timestamp() - timestamp.timestamp();
            if age < self.cache_ttl_seconds as i64 {
                tracing::debug!("Retrieved image embedding from cache");
                return Ok(embedding.clone());
            }
        }

        let model = self.model.as_ref().ok_or_else(|| NLPError::ModelLoading {
            message: "Image embedding model not initialized".to_string(),
        })?;

        // Load image using the image crate
        let img = image::load_from_memory(image_data).map_err(|e| NLPError::ProcessingError {
            message: format!("Failed to load image: {}", e),
        })?;

        // Save image to temporary file (fastembed works with file paths)
        let temp_dir = std::env::temp_dir();
        let temp_file = temp_dir.join(format!("temp_img_{}.png", uuid::Uuid::new_v4()));

        img.save(&temp_file)
            .map_err(|e| NLPError::ProcessingError {
                message: format!("Failed to save temporary image: {}", e),
            })?;

        // Generate embeddings using fastembed
        let embeddings =
            model
                .embed(vec![temp_file.clone()], None)
                .map_err(|e| NLPError::ProcessingError {
                    message: format!("Failed to generate image embedding: {}", e),
                })?;

        // Clean up temporary file
        let _ = std::fs::remove_file(&temp_file);

        let embedding = embeddings
            .into_iter()
            .next()
            .ok_or_else(|| NLPError::ProcessingError {
                message: "No embedding generated".to_string(),
            })?;

        // Cache the result
        self.cache
            .insert(cache_key, (embedding.clone(), Utc::now()));

        let duration = start_time.elapsed();
        tracing::debug!("Image embedding generated in {:?}", duration);

        Ok(embedding)
    }

    /// Batch generate embeddings for multiple images
    pub async fn batch_embeddings(
        &self,
        image_data_list: &[&[u8]],
    ) -> Result<Vec<Vec<f32>>, NLPError> {
        let _timer = Timer::new("batch_image_embedding_generation");

        let model = self.model.as_ref().ok_or_else(|| NLPError::ModelLoading {
            message: "Image embedding model not initialized".to_string(),
        })?;

        // Load all images and save to temporary files
        let temp_dir = std::env::temp_dir();
        let mut temp_files = Vec::new();

        for (i, image_data) in image_data_list.iter().enumerate() {
            let img =
                image::load_from_memory(image_data).map_err(|e| NLPError::ProcessingError {
                    message: format!("Failed to load image: {}", e),
                })?;

            let temp_file =
                temp_dir.join(format!("temp_batch_img_{}_{}.png", uuid::Uuid::new_v4(), i));
            img.save(&temp_file)
                .map_err(|e| NLPError::ProcessingError {
                    message: format!("Failed to save temporary image: {}", e),
                })?;
            temp_files.push(temp_file);
        }

        // Generate embeddings in batch
        let embeddings =
            model
                .embed(temp_files.clone(), None)
                .map_err(|e| NLPError::ProcessingError {
                    message: format!("Failed to generate batch image embeddings: {}", e),
                })?;

        // Clean up temporary files
        for temp_file in temp_files {
            let _ = std::fs::remove_file(temp_file);
        }

        tracing::debug!("Generated {} image embeddings in batch", embeddings.len());
        Ok(embeddings)
    }

    /// Get embedding dimensions
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> (usize, usize) {
        (self.cache.len(), 1000) // Current size, capacity
    }

    /// Clear the embedding cache
    pub fn clear_cache(&self) {
        self.cache.clear();
        tracing::debug!("Cleared image embedding cache");
    }

    /// Compute cache key for image data
    fn compute_cache_key(&self, image_data: &[u8]) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        image_data.hash(&mut hasher);
        format!("img_{:x}", hasher.finish())
    }
}

/// Image metadata extractor
#[cfg(feature = "multimodal")]
pub struct ImageMetadataExtractor;

#[cfg(feature = "multimodal")]
impl ImageMetadataExtractor {
    /// Extract comprehensive metadata from image
    pub async fn extract_metadata(image_data: &[u8]) -> Result<ImageMetadata, NLPError> {
        let _timer = Timer::new("image_metadata_extraction");
        let start_time = Instant::now();

        // Load image to get basic information
        let img = image::load_from_memory(image_data).map_err(|e| NLPError::ProcessingError {
            message: format!("Failed to load image: {}", e),
        })?;

        let dimensions = (img.width(), img.height());
        let format = Self::detect_format(image_data)?;

        // Extract EXIF data
        let (timestamp, gps_coordinates, camera_info, orientation) =
            Self::extract_exif_data(image_data)?;

        let processing_time = start_time.elapsed().as_millis() as u64;

        Ok(ImageMetadata {
            dimensions,
            format,
            file_size: image_data.len(),
            timestamp,
            gps_coordinates,
            camera_info,
            color_space: Some(Self::detect_color_space(&img)),
            orientation,
            processing_time_ms: processing_time,
        })
    }

    /// Detect image format from data
    fn detect_format(image_data: &[u8]) -> Result<String, NLPError> {
        let format = image::guess_format(image_data).map_err(|e| NLPError::ProcessingError {
            message: format!("Failed to detect image format: {}", e),
        })?;

        Ok(match format {
            image::ImageFormat::Jpeg => "JPEG".to_string(),
            image::ImageFormat::Png => "PNG".to_string(),
            image::ImageFormat::Gif => "GIF".to_string(),
            image::ImageFormat::WebP => "WebP".to_string(),
            image::ImageFormat::Tiff => "TIFF".to_string(),
            image::ImageFormat::Bmp => "BMP".to_string(),
            _ => "Unknown".to_string(),
        })
    }

    /// Extract EXIF metadata
    fn extract_exif_data(image_data: &[u8]) -> ExifResult {
        let mut cursor = std::io::Cursor::new(image_data);

        match exif::Reader::new().read_from_container(&mut cursor) {
            Ok(exif_data) => {
                let timestamp = Self::extract_timestamp(&exif_data);
                let gps_coordinates = Self::extract_gps_coordinates(&exif_data);
                let camera_info = Self::extract_camera_info(&exif_data);
                let orientation = Self::extract_orientation(&exif_data);

                Ok((timestamp, gps_coordinates, camera_info, orientation))
            }
            Err(_) => {
                // No EXIF data available
                Ok((None, None, None, None))
            }
        }
    }

    /// Extract timestamp from EXIF data
    fn extract_timestamp(exif_data: &exif::Exif) -> Option<DateTime<Utc>> {
        exif_data
            .get_field(exif::Tag::DateTime, exif::In::PRIMARY)
            .and_then(|field| {
                if let exif::Value::Ascii(ref vec) = field.value {
                    if let Some(ascii_str) = vec.first() {
                        let datetime_str = String::from_utf8_lossy(ascii_str);
                        // Parse EXIF datetime format: "YYYY:MM:DD HH:MM:SS"
                        chrono::NaiveDateTime::parse_from_str(&datetime_str, "%Y:%m:%d %H:%M:%S")
                            .ok()
                            .map(|naive_dt| {
                                DateTime::<Utc>::from_naive_utc_and_offset(naive_dt, Utc)
                            })
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
    }

    /// Extract GPS coordinates from EXIF data
    fn extract_gps_coordinates(exif_data: &exif::Exif) -> Option<(f64, f64)> {
        let lat = exif_data.get_field(exif::Tag::GPSLatitude, exif::In::PRIMARY)?;
        let lat_ref = exif_data.get_field(exif::Tag::GPSLatitudeRef, exif::In::PRIMARY)?;
        let lon = exif_data.get_field(exif::Tag::GPSLongitude, exif::In::PRIMARY)?;
        let lon_ref = exif_data.get_field(exif::Tag::GPSLongitudeRef, exif::In::PRIMARY)?;

        let latitude = Self::parse_gps_coordinate(&lat.value, &lat_ref.value)?;
        let longitude = Self::parse_gps_coordinate(&lon.value, &lon_ref.value)?;

        Some((latitude, longitude))
    }

    /// Parse GPS coordinate from EXIF rational values
    fn parse_gps_coordinate(coord: &exif::Value, ref_val: &exif::Value) -> Option<f64> {
        if let (exif::Value::Rational(ref rationals), exif::Value::Ascii(ref ref_bytes)) =
            (coord, ref_val)
        {
            if rationals.len() >= 3 && !ref_bytes.is_empty() {
                let degrees = rationals[0].to_f64();
                let minutes = rationals[1].to_f64();
                let seconds = rationals[2].to_f64();

                let decimal = degrees + minutes / 60.0 + seconds / 3600.0;

                // Apply hemisphere reference
                let ref_char = ref_bytes[0][0] as char;
                let result = if ref_char == 'S' || ref_char == 'W' {
                    -decimal
                } else {
                    decimal
                };

                Some(result)
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Extract camera information from EXIF data
    fn extract_camera_info(exif_data: &exif::Exif) -> Option<CameraInfo> {
        let make = Self::extract_string_field(exif_data, exif::Tag::Make);
        let model = Self::extract_string_field(exif_data, exif::Tag::Model);
        let lens_model = Self::extract_string_field(exif_data, exif::Tag::LensModel);

        let focal_length = Self::extract_rational_field(exif_data, exif::Tag::FocalLength);
        let aperture = Self::extract_rational_field(exif_data, exif::Tag::FNumber);
        let iso = Self::extract_integer_field(exif_data, exif::Tag::PhotographicSensitivity);
        let exposure_time = Self::extract_rational_field(exif_data, exif::Tag::ExposureTime);

        let flash = exif_data
            .get_field(exif::Tag::Flash, exif::In::PRIMARY)
            .and_then(|field| {
                if let exif::Value::Short(ref values) = field.value {
                    values.first().map(|&val| val & 1 == 1) // Check flash fired bit
                } else {
                    None
                }
            });

        // Return Some only if we have at least some camera info
        if make.is_some() || model.is_some() || lens_model.is_some() {
            Some(CameraInfo {
                make,
                model,
                lens_model,
                focal_length,
                aperture,
                iso,
                exposure_time,
                flash,
            })
        } else {
            None
        }
    }

    /// Extract orientation from EXIF data
    fn extract_orientation(exif_data: &exif::Exif) -> Option<u8> {
        exif_data
            .get_field(exif::Tag::Orientation, exif::In::PRIMARY)
            .and_then(|field| {
                if let exif::Value::Short(ref values) = field.value {
                    values.first().map(|&val| val as u8)
                } else {
                    None
                }
            })
    }

    /// Helper to extract string field from EXIF
    fn extract_string_field(exif_data: &exif::Exif, tag: exif::Tag) -> Option<String> {
        exif_data
            .get_field(tag, exif::In::PRIMARY)
            .and_then(|field| {
                if let exif::Value::Ascii(ref vec) = field.value {
                    vec.first().map(|ascii_bytes| {
                        String::from_utf8_lossy(ascii_bytes)
                            .trim_end_matches('\0')
                            .to_string()
                    })
                } else {
                    None
                }
            })
    }

    /// Helper to extract rational field from EXIF
    fn extract_rational_field(exif_data: &exif::Exif, tag: exif::Tag) -> Option<f32> {
        exif_data
            .get_field(tag, exif::In::PRIMARY)
            .and_then(|field| {
                if let exif::Value::Rational(ref rationals) = field.value {
                    rationals.first().map(|rational| rational.to_f64() as f32)
                } else {
                    None
                }
            })
    }

    /// Helper to extract integer field from EXIF
    fn extract_integer_field(exif_data: &exif::Exif, tag: exif::Tag) -> Option<u32> {
        exif_data
            .get_field(tag, exif::In::PRIMARY)
            .and_then(|field| match field.value {
                exif::Value::Short(ref values) => values.first().map(|&val| val as u32),
                exif::Value::Long(ref values) => values.first().copied(),
                _ => None,
            })
    }

    /// Detect color space of image
    fn detect_color_space(img: &image::DynamicImage) -> String {
        match img {
            image::DynamicImage::ImageRgb8(_) => "RGB".to_string(),
            image::DynamicImage::ImageRgba8(_) => "RGBA".to_string(),
            image::DynamicImage::ImageLuma8(_) => "Grayscale".to_string(),
            image::DynamicImage::ImageLumaA8(_) => "Grayscale+Alpha".to_string(),
            _ => "Unknown".to_string(),
        }
    }
}
