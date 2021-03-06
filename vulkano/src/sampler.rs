// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! How to retreive data from an image within a shader.
//! 
//! This module contains a struct named `Sampler` which describes how to get pixel data from
//! a texture.
//!
use std::error;
use std::fmt;
use std::mem;
use std::ptr;
use std::sync::Arc;

use device::Device;
use Error;
use OomError;
use VulkanObject;
use VulkanPointers;
use check_errors;
use vk;

/// Describes how to retreive data from an image within a shader.
pub struct Sampler {
    sampler: vk::Sampler,
    device: Arc<Device>,
}

// TODO: what's the story with VK_KHR_mirror_clamp_to_edge? Is it an extension or is it core?

impl Sampler {
    /// Creates a new `Sampler` with the given behavior.
    ///
    /// # Panic
    ///
    /// - Panicks if `max_anisotropy < 1.0`.
    /// - Panicks if `min_lod > max_lod`.
    ///
    pub fn new(device: &Arc<Device>, mag_filter: Filter, min_filter: Filter,
               mipmap_mode: MipmapMode, address_u: SamplerAddressMode,
               address_v: SamplerAddressMode, address_w: SamplerAddressMode, mip_lod_bias: f32,
               max_anisotropy: f32, min_lod: f32, max_lod: f32)
               -> Result<Arc<Sampler>, SamplerCreationError>
    {
        assert!(max_anisotropy >= 1.0);
        assert!(min_lod <= max_lod);

        if max_anisotropy > 1.0 {
            if !device.enabled_features().sampler_anisotropy {
                return Err(SamplerCreationError::SamplerAnisotropyFeatureNotEnabled);
            }

            let limit = device.physical_device().limits().max_sampler_anisotropy();
            if max_anisotropy > limit {
                return Err(SamplerCreationError::AnisotropyLimitExceeded {
                    requested: max_anisotropy,
                    maximum: limit,
                });
            }
        }

        {
            let limit = device.physical_device().limits().max_sampler_lod_bias();
            if mip_lod_bias > limit {
                return Err(SamplerCreationError::MipLodBiasLimitExceeded {
                    requested: mip_lod_bias,
                    maximum: limit,
                });
            }
        }

        let vk = device.pointers();

        let sampler = unsafe {
            let infos = vk::SamplerCreateInfo {
                sType: vk::STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,   // reserved
                magFilter: mag_filter as u32,
                minFilter: min_filter as u32,
                mipmapMode: mipmap_mode as u32,
                addressModeU: address_u as u32,
                addressModeV: address_v as u32,
                addressModeW: address_w as u32,
                mipLodBias: mip_lod_bias,
                anisotropyEnable: if max_anisotropy > 1.0 { vk::TRUE } else { vk::FALSE },
                maxAnisotropy: max_anisotropy,
                compareEnable: 0,       // FIXME: 
                compareOp: 0,       // FIXME: 
                minLod: min_lod,
                maxLod: max_lod,
                borderColor: 0,     // FIXME: 
                unnormalizedCoordinates: vk::FALSE,
            };

            let mut output = mem::uninitialized();
            try!(check_errors(vk.CreateSampler(device.internal_object(), &infos,
                                               ptr::null(), &mut output)));
            output
        };

        Ok(Arc::new(Sampler {
            sampler: sampler,
            device: device.clone(),
        }))
    }

    /// Creates a sampler with unnormalized coordinates. This means that texture coordinates won't
    /// range between `0.0` and `1.0` but use plain pixel offsets.
    ///
    /// Using an unnormalized sampler adds a few restrictions:
    ///
    /// - It can only be used with non-array 1D or 2D images.
    /// - It can only be used with images with a single mipmap.
    /// - Projection and offsets can't be used by shaders. Only the first mipmap can be accessed.
    ///
    pub fn unnormalized(device: &Arc<Device>, filter: Filter,
                        address_u: UnnormalizedSamplerAddressMode,
                        address_v: UnnormalizedSamplerAddressMode)
                        -> Result<Arc<Sampler>, SamplerCreationError>
    {
        let vk = device.pointers();

        let sampler = unsafe {
            let infos = vk::SamplerCreateInfo {
                sType: vk::STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,   // reserved
                magFilter: filter as u32,
                minFilter: filter as u32,
                mipmapMode: vk::SAMPLER_MIPMAP_MODE_NEAREST,
                addressModeU: address_u as u32,
                addressModeV: address_v as u32,
                addressModeW: vk::SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,       // unused by the impl
                mipLodBias: 0.0,
                anisotropyEnable: vk::FALSE,
                maxAnisotropy: 0.0,
                compareEnable: vk::FALSE,
                compareOp: vk::COMPARE_OP_NEVER,
                minLod: 0.0,
                maxLod: 0.0,
                borderColor: 0,     // FIXME: 
                unnormalizedCoordinates: vk::TRUE,
            };

            let mut output = mem::uninitialized();
            try!(check_errors(vk.CreateSampler(device.internal_object(), &infos,
                                               ptr::null(), &mut output)));
            output
        };

        Ok(Arc::new(Sampler {
            sampler: sampler,
            device: device.clone(),
        }))
    }
}

unsafe impl VulkanObject for Sampler {
    type Object = vk::Sampler;

    #[inline]
    fn internal_object(&self) -> vk::Sampler {
        self.sampler
    }
}

impl Drop for Sampler {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let vk = self.device.pointers();
            vk.DestroySampler(self.device.internal_object(), self.sampler, ptr::null());
        }
    }
}

/// Describes how the color of each pixel should be determined.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum Filter {
    /// The four pixels whose center surround the requested coordinates are taken, then their
    /// values are interpolated.
    Linear = vk::FILTER_LINEAR,

    /// The pixel whose center is nearest to the requested coordinates is taken from the source
    /// and its value is returned as-is.
    Nearest = vk::FILTER_NEAREST,
}

/// Describes which mipmap from the source to use.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum MipmapMode {
    /// Use the mipmap whose dimensions are the nearest to the dimensions of the destination.
    Nearest = vk::SAMPLER_MIPMAP_MODE_NEAREST,

    /// Take the two mipmaps whose dimensions are immediately inferior and superior to the
    /// dimensions of the destination, calculate the value for both, and interpolate them.
    Linear = vk::SAMPLER_MIPMAP_MODE_LINEAR,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum SamplerAddressMode {
    Repeat = vk::SAMPLER_ADDRESS_MODE_REPEAT,
    MirroredRepeat = vk::SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT,
    ClampToEdge = vk::SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
    ClampToBorder = vk::SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
    MirrorClampToEdge = vk::SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum UnnormalizedSamplerAddressMode {
    ClampToEdge = vk::SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
    ClampToBorder = vk::SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
}

/// Error that can happen when creating an instance.
#[derive(Clone, Debug, PartialEq)]
pub enum SamplerCreationError {
    /// Not enough memory.
    OomError(OomError),

    /// Too many sampler objects have been created. You must destroy some before creating new ones.
    /// Note the specs guarantee that at least 4000 samplers can exist simultaneously.
    TooManyObjects,

    /// Using an anisotropy superior to 1.0 requires enabling the `sampler_anisotropy` feature when
    /// creating the device.
    SamplerAnisotropyFeatureNotEnabled,

    /// The requested anisotropy level exceeds the device's limits.
    AnisotropyLimitExceeded { requested: f32, maximum: f32 },

    /// The requested mip lod bias exceeds the device's limits.
    MipLodBiasLimitExceeded { requested: f32, maximum: f32 },
}

impl error::Error for SamplerCreationError {
    #[inline]
    fn description(&self) -> &str {
        match *self {
            SamplerCreationError::OomError(_) => "not enough memory available",
            SamplerCreationError::TooManyObjects => "too many simultaneous sampler objects",
            SamplerCreationError::SamplerAnisotropyFeatureNotEnabled => "the `sampler_anisotropy` \
                                                                         feature is not enabled",
            SamplerCreationError::AnisotropyLimitExceeded { .. } => "anisotropy limit exceeded",
            SamplerCreationError::MipLodBiasLimitExceeded { .. } => "mip lod bias limit exceeded",
        }
    }

    #[inline]
    fn cause(&self) -> Option<&error::Error> {
        match *self {
            SamplerCreationError::OomError(ref err) => Some(err),
            _ => None
        }
    }
}

impl fmt::Display for SamplerCreationError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "{}", error::Error::description(self))
    }
}

impl From<OomError> for SamplerCreationError {
    #[inline]
    fn from(err: OomError) -> SamplerCreationError {
        SamplerCreationError::OomError(err)
    }
}

impl From<Error> for SamplerCreationError {
    #[inline]
    fn from(err: Error) -> SamplerCreationError {
        match err {
            err @ Error::OutOfHostMemory => SamplerCreationError::OomError(OomError::from(err)),
            err @ Error::OutOfDeviceMemory => SamplerCreationError::OomError(OomError::from(err)),
            Error::TooManyObjects => SamplerCreationError::TooManyObjects,
            _ => panic!("unexpected error: {:?}", err)
        }
    }
}

#[cfg(test)]
mod tests {
    use sampler;

    #[test]
    fn create_regular() {
        let (device, queue) = gfx_dev_and_queue!();

        let _ = sampler::Sampler::new(&device, sampler::Filter::Linear, sampler::Filter::Linear,
                                      sampler::MipmapMode::Nearest,
                                      sampler::SamplerAddressMode::Repeat,
                                      sampler::SamplerAddressMode::Repeat,
                                      sampler::SamplerAddressMode::Repeat, 1.0, 1.0,
                                      0.0, 2.0).unwrap();
    }

    #[test]
    fn create_unnormalized() {
        let (device, queue) = gfx_dev_and_queue!();

        let _ = sampler::Sampler::unnormalized(&device, sampler::Filter::Linear,
                                               sampler::UnnormalizedSamplerAddressMode::ClampToEdge,
                                               sampler::UnnormalizedSamplerAddressMode::ClampToEdge)
                                               .unwrap();
    }

    #[test]
    #[should_panic]
    fn min_lod_inferior() {
        let (device, queue) = gfx_dev_and_queue!();

        let _ = sampler::Sampler::new(&device, sampler::Filter::Linear, sampler::Filter::Linear,
                                      sampler::MipmapMode::Nearest,
                                      sampler::SamplerAddressMode::Repeat,
                                      sampler::SamplerAddressMode::Repeat,
                                      sampler::SamplerAddressMode::Repeat, 1.0, 1.0, 5.0, 2.0);
    }

    #[test]
    #[should_panic]
    fn max_anisotropy() {
        let (device, queue) = gfx_dev_and_queue!();

        let _ = sampler::Sampler::new(&device, sampler::Filter::Linear, sampler::Filter::Linear,
                                      sampler::MipmapMode::Nearest,
                                      sampler::SamplerAddressMode::Repeat,
                                      sampler::SamplerAddressMode::Repeat,
                                      sampler::SamplerAddressMode::Repeat, 1.0, 0.5, 0.0, 2.0);
    }

    #[test]
    fn anisotropy_feature() {
        let (device, queue) = gfx_dev_and_queue!();

        let r = sampler::Sampler::new(&device, sampler::Filter::Linear, sampler::Filter::Linear,
                                      sampler::MipmapMode::Nearest,
                                      sampler::SamplerAddressMode::Repeat,
                                      sampler::SamplerAddressMode::Repeat,
                                      sampler::SamplerAddressMode::Repeat, 1.0, 2.0, 0.0, 2.0);

        match r {
            Err(sampler::SamplerCreationError::SamplerAnisotropyFeatureNotEnabled) => (),
            _ => panic!()
        }
    }

    #[test]
    fn anisotropy_limit() {
        let (device, queue) = gfx_dev_and_queue!(sampler_anisotropy);

        let r = sampler::Sampler::new(&device, sampler::Filter::Linear, sampler::Filter::Linear,
                                      sampler::MipmapMode::Nearest,
                                      sampler::SamplerAddressMode::Repeat,
                                      sampler::SamplerAddressMode::Repeat,
                                      sampler::SamplerAddressMode::Repeat, 1.0, 100000000.0, 0.0,
                                      2.0);

        match r {
            Err(sampler::SamplerCreationError::AnisotropyLimitExceeded { .. }) => (),
            _ => panic!()
        }
    }

    #[test]
    fn mip_lod_bias_limit() {
        let (device, queue) = gfx_dev_and_queue!();

        let r = sampler::Sampler::new(&device, sampler::Filter::Linear, sampler::Filter::Linear,
                                      sampler::MipmapMode::Nearest,
                                      sampler::SamplerAddressMode::Repeat,
                                      sampler::SamplerAddressMode::Repeat,
                                      sampler::SamplerAddressMode::Repeat, 100000000.0, 1.0, 0.0,
                                      2.0);

        match r {
            Err(sampler::SamplerCreationError::MipLodBiasLimitExceeded { .. }) => (),
            _ => panic!()
        }
    }
}
