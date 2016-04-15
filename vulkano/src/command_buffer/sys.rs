// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::error;
use std::fmt;
use std::hash;
use std::hash::BuildHasherDefault;
use std::mem;
use std::ops::Range;
use std::ptr;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;
use std::u64;
use fnv::FnvHasher;
use smallvec::SmallVec;

use buffer::Buffer;
use buffer::BufferSlice;
use buffer::TypedBuffer;
use buffer::traits::AccessRange as BufferAccessRange;
use command_buffer::CommandBufferPool;
use command_buffer::DrawIndirectCommand;
use command_buffer::DynamicState;
use descriptor::descriptor_set::DescriptorSetsCollection;
use descriptor::PipelineLayout;
use device::Queue;
use format::ClearValue;
use format::FormatDesc;
use format::FormatTy;
use format::PossibleFloatFormatDesc;
use framebuffer::RenderPass;
use framebuffer::RenderPassDesc;
use framebuffer::Framebuffer;
use framebuffer::Subpass;
use image::Image;
use image::ImageView;
use image::sys::Layout as ImageLayout;
use image::traits::ImageClearValue;
use image::traits::ImageContent;
use image::traits::AccessRange as ImageAccessRange;
use pipeline::ComputePipeline;
use pipeline::GraphicsPipeline;
use pipeline::input_assembly::Index;
use pipeline::vertex::Definition as VertexDefinition;
use pipeline::vertex::Source as VertexSource;
use sync::Fence;
use sync::FenceWaitError;
use sync::Semaphore;

use device::Device;
use OomError;
use SynchronizedVulkanObject;
use VulkanObject;
use VulkanPointers;
use check_errors;
use vk;

pub struct UnsafeCommandBufferBuilder {
    cmd: Option<vk::CommandBuffer>,
    device: Arc<Device>,
    pool: Arc<CommandBufferPool>,

    // List of resources that must be kept alive because they are used by this command buffer.
    keep_alive: Vec<Arc<KeepAlive>>,

    // Current pipeline object binded to the graphics bind point. Includes all staging commands.
    current_graphics_pipeline: Option<vk::Pipeline>,

    // Current pipeline object binded to the compute bind point. Includes all staging commands.
    current_compute_pipeline: Option<vk::Pipeline>,

    // Current state of the dynamic state within the command buffer. Includes all staging commands.
    current_dynamic_state: DynamicState,

    // True if we are a secondary command buffer.
    secondary_cb: bool,

    // True if we are within a render pass.
    within_render_pass: bool,
}

impl UnsafeCommandBufferBuilder {
    /*/// Creates a new builder.
    pub fn new<R>(pool: &Arc<CommandBufferPool>, secondary: bool, secondary_cont: Option<Subpass<R>>,
                  secondary_cont_fb: Option<&Arc<Framebuffer<R>>>)
                  -> Result<UnsafeCommandBufferBuilder, OomError>
        where R: RenderPass + 'static + Send + Sync
    {
        let device = pool.device();
        let vk = device.pointers();

        let pool_obj = pool.internal_object_guard();

        let cmd = unsafe {
            let infos = vk::CommandBufferAllocateInfo {
                sType: vk::STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                pNext: ptr::null(),
                commandPool: *pool_obj,
                level: if secondary {
                    assert!(secondary_cont.is_some());
                    vk::COMMAND_BUFFER_LEVEL_SECONDARY
                } else {
                    vk::COMMAND_BUFFER_LEVEL_PRIMARY
                },
                // vulkan can allocate multiple command buffers at once, hence the 1
                commandBufferCount: 1,
            };

            let mut output = mem::uninitialized();
            try!(check_errors(vk.AllocateCommandBuffers(device.internal_object(), &infos,
                                                        &mut output)));
            output
        };

        let mut keep_alive = Vec::new();

        unsafe {
            // TODO: one time submit
            let flags = vk::COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT |     // TODO:
                        if secondary_cont.is_some() { vk::COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT } else { 0 };

            let (rp, sp) = if let Some(ref sp) = secondary_cont {
                keep_alive.push(sp.render_pass().clone() as Arc<_>);
                (sp.render_pass().render_pass().internal_object(), sp.index())
            } else {
                (0, 0)
            };

            let framebuffer = if let Some(fb) = secondary_cont_fb {
                keep_alive.push(fb.clone() as Arc<_>);
                fb.internal_object()
            } else {
                0
            };

            let inheritance = vk::CommandBufferInheritanceInfo {
                sType: vk::STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO,
                pNext: ptr::null(),
                renderPass: rp,
                subpass: sp,
                framebuffer: framebuffer,
                occlusionQueryEnable: 0,            // TODO:
                queryFlags: 0,          // TODO:
                pipelineStatistics: 0,          // TODO:
            };

            let infos = vk::CommandBufferBeginInfo {
                sType: vk::STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                pNext: ptr::null(),
                flags: flags,
                pInheritanceInfo: &inheritance,
            };

            try!(check_errors(vk.BeginCommandBuffer(cmd, &infos)));
        }

        Ok(UnsafeCommandBufferBuilder {
            device: device.clone(),
            pool: pool.clone(),
            cmd: Some(cmd),
            keep_alive: keep_alive,
            current_graphics_pipeline: None,
            current_compute_pipeline: None,
            current_dynamic_state: DynamicState::none(),
        })
    }*/

    /// Adds a command that copies regions between a source and a destination buffer. Does not
    /// check the type of the content, contrary to `copy_buffer`.
    ///
    /// Regions whose size is 0 are automatically ignored. If no region was passed or if all
    /// regions have a size of 0, then no command is added to the command buffer.
    ///
    /// # Panic
    ///
    /// - Panicks if one of the buffers was not allocated with the same device as this command
    ///   buffer.
    ///
    pub fn copy_buffer_untyped<Bs, Bd, I>(mut self, src: &Arc<Bs>, dest: &Arc<Bd>, regions: I)
                                          -> Result<UnsafeCommandBufferBuilder, BufferCopyError>
        where Bs: Buffer + Send + Sync + 'static,
              Bd: Buffer + Send + Sync + 'static,
              I: IntoIterator<Item = BufferCopyRegion>
    {
        unsafe {
            // Various safety checks.
            if self.within_render_pass { return Err(BufferCopyError::ForbiddenWithinRenderPass); }
            assert_eq!(src.inner_buffer().device().internal_object(),
                       self.pool.device().internal_object());
            assert_eq!(dest.inner_buffer().device().internal_object(),
                       self.pool.device().internal_object());
            if !src.inner_buffer().usage_transfer_src() ||
               !dest.inner_buffer().usage_transfer_dest()
            {
                return Err(BufferCopyError::WrongUsageFlag);
            }

            // Building the list of regions.
            let regions: SmallVec<[_; 4]> = {
                let mut res = SmallVec::new();
                for region in regions.into_iter() {
                    if region.source_offset + region.size > src.size() {
                        return Err(BufferCopyError::OutOfRange);
                    }
                    if region.destination_offset + region.size > dest.size() {
                        return Err(BufferCopyError::OutOfRange);
                    }
                    if region.size == 0 { continue; }

                    res.push(vk::BufferCopy {
                        srcOffset: region.source_offset as vk::DeviceSize,
                        dstOffset: region.destination_offset as vk::DeviceSize,
                        size: region.size as vk::DeviceSize,
                    });
                }
                res
            };

            // Vulkan requires that the number of regions must always be >= 1.
            if regions.is_empty() { return Ok(self); }

            // Checking for overlaps.
            for r1 in 0 .. regions.len() {
                for r2 in (r1 + 1) .. regions.len() {
                    let r1 = &regions[r1];
                    let r2 = &regions[r2];

                    if r1.srcOffset <= r2.srcOffset && r1.srcOffset + r1.size >= r2.srcOffset {
                        return Err(BufferCopyError::OverlappingRegions);
                    }
                    if r2.srcOffset <= r1.srcOffset && r2.srcOffset + r2.size >= r1.srcOffset {
                        return Err(BufferCopyError::OverlappingRegions);
                    }
                    if r1.dstOffset <= r2.dstOffset && r1.dstOffset + r1.size >= r2.dstOffset {
                        return Err(BufferCopyError::OverlappingRegions);
                    }
                    if r2.dstOffset <= r1.dstOffset && r2.dstOffset + r2.size >= r1.dstOffset {
                        return Err(BufferCopyError::OverlappingRegions);
                    }

                    if src.inner_buffer().internal_object() ==
                       dest.inner_buffer().internal_object()
                    {
                        if r1.srcOffset <= r2.dstOffset && r1.srcOffset + r1.size >= r2.dstOffset {
                            return Err(BufferCopyError::OverlappingRegions);
                        }
                        if r2.srcOffset <= r1.dstOffset && r2.srcOffset + r2.size >= r1.dstOffset {
                            return Err(BufferCopyError::OverlappingRegions);
                        }
                    }
                }
            }

            // Now adding the command.
            self.keep_alive.push(src.clone());
            self.keep_alive.push(dest.clone());

            {
                let vk = self.device.pointers();
                let cmd = self.cmd.clone().unwrap();
                vk.CmdCopyBuffer(cmd, src.inner_buffer().internal_object(),
                                 dest.inner_buffer().internal_object(), regions.len() as u32,
                                 regions.as_ptr());
            }

            Ok(self)
        }
    }
}

/// A copy between two buffers.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct BufferCopyRegion {
    /// Offset of the first byte to read from the source buffer.
    pub source_offset: usize,
    /// Offset of the first byte to write to the destination buffer.
    pub destination_offset: usize,
    /// Size in bytes of the copy.
    pub size: usize,
}

macro_rules! error_ty {
    ($err_name:ident => $doc:expr, $($member:ident => $desc:expr,)*) => {
        #[doc = $doc]
        #[derive(Clone, Debug, PartialEq, Eq)]
        pub enum $err_name {
            $(
                #[doc = $desc]
                $member
            ),*
        }

        impl error::Error for $err_name {
            #[inline]
            fn description(&self) -> &str {
                match *self {
                    $(
                        $err_name::$member => $desc,
                    )*
                }
            }
        }

        impl fmt::Display for $err_name {
            #[inline]
            fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
                write!(fmt, "{}", error::Error::description(self))
            }
        }
    };
}

error_ty!{BufferCopyError => "Error that can happen when copying between buffers.",
    ForbiddenWithinRenderPass => "can't copy buffers from within a render pass",
    OutOfRange => "one of regions is out of range of the buffer",
    WrongUsageFlag => "one of the buffers doesn't have the correct usage flag",
    OverlappingRegions => "some regions are overlapping",
}

/// Dummy trait that is implemented on everything and that allows us to keep Arcs alive.
trait KeepAlive: 'static + Send + Sync {}
impl<T> KeepAlive for T where T: 'static + Send + Sync {}
