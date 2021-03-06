// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::ffi::CStr;
use std::ptr;
use std::vec::IntoIter;

//use alloc::Alloc;
use check_errors;
use OomError;
use vk;
use instance::loader;
use version::Version;

/// Queries the list of layers that are available when creating an instance.
pub fn layers_list() -> Result<LayersIterator, OomError> {
    unsafe {
        let entry_points = loader::entry_points().unwrap();     // TODO: return proper error

        let mut num = 0;
        try!(check_errors(entry_points.EnumerateInstanceLayerProperties(&mut num, ptr::null_mut())));

        let mut layers: Vec<vk::LayerProperties> = Vec::with_capacity(num as usize);
        try!(check_errors(entry_points.EnumerateInstanceLayerProperties(&mut num,
                                                                        layers.as_mut_ptr())));
        layers.set_len(num as usize);

        Ok(LayersIterator {
            iter: layers.into_iter()
        })
    }
}

/// Properties of an available layer.
pub struct LayerProperties {
    props: vk::LayerProperties,
}

impl LayerProperties {
    /// Returns the name of the layer.
    #[inline]
    pub fn name(&self) -> &str {
        unsafe { CStr::from_ptr(self.props.layerName.as_ptr()).to_str().unwrap() }
    }

    /// Returns a description of the layer.
    #[inline]
    pub fn description(&self) -> &str {
        unsafe { CStr::from_ptr(self.props.description.as_ptr()).to_str().unwrap() }
    }

    /// Returns the version of Vulkan supported by this layer.
    #[inline]
    pub fn vulkan_version(&self) -> Version {
        Version::from_vulkan_version(self.props.specVersion)
    }

    /// Returns an implementation-specific version number for this layer.
    #[inline]
    pub fn implementation_version(&self) -> u32 {
        self.props.implementationVersion
    }
}

/// Iterator that produces the list of layers that are available.
// TODO: #[derive(Debug, Clone)]
pub struct LayersIterator {
    iter: IntoIter<vk::LayerProperties>
}

impl Iterator for LayersIterator {
    type Item = LayerProperties;

    #[inline]
    fn next(&mut self) -> Option<LayerProperties> {
        self.iter.next().map(|p| LayerProperties { props: p })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl ExactSizeIterator for LayersIterator {
}

#[cfg(test)]
mod tests {
    use instance;

    #[test]
    fn layers_list() {
        let mut list = instance::layers_list().unwrap();
        while let Some(_) = list.next() {}
    }
}
