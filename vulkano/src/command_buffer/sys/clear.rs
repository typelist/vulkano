// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::error;
use std::fmt;
use std::mem;
use std::sync::Arc;
use smallvec::SmallVec;

use buffer::Buffer;
use buffer::BufferSlice;
use buffer::TypedBuffer;
use command_buffer::sys::UnsafeCommandBufferBuilder;

use VulkanObject;
use VulkanPointers;
use vk;

impl UnsafeCommandBufferBuilder {
    /// Adds a command that fills a buffer with some data. The data is a u32 whose value will be
    /// repeatidely written in the buffer.
    ///
    /// If the size of the slice is 0, no command is added.
    ///
    /// # Panic
    ///
    /// - Panicks if the buffer was not allocated with the same device as this command buffer.
    ///
    pub fn fill_buffer_untyped<'a, S, T: ?Sized, B>(mut self, buffer: S, data: u32)
                                                    -> Result<UnsafeCommandBufferBuilder,
                                                              BufferFillError>
        where S: Into<BufferSlice<'a, T, B>>,
              B: Buffer + Send + Sync + 'static
    {
        unsafe {
            let buffer = buffer.into();

            // Performing checks.
            if self.within_render_pass { return Err(BufferFillError::ForbiddenWithinRenderPass); }
            assert_eq!(buffer.buffer().inner_buffer().device().internal_object(),
                       self.pool.device().internal_object());
            if !buffer.buffer().inner_buffer().usage_transfer_src() {
                return Err(BufferFillError::WrongUsageFlag);
            }
            if (buffer.offset() % 4) != 0 || (buffer.size() % 4) != 0 {
                return Err(BufferFillError::WrongAlignment);
            }
            if buffer.size() == 0 { return Ok(self); }
            if !self.pool.queue_family().supports_graphics() &&
               !self.pool.queue_family().supports_compute()
            {
                return Err(BufferFillError::NotSupportedByQueueFamily);
            }

            // Adding the command.
            self.keep_alive.push(buffer.buffer().clone());

            {
                let vk = self.device.pointers();
                let cmd = self.cmd.clone().unwrap();
                vk.CmdFillBuffer(cmd, buffer.buffer().inner_buffer().internal_object(),
                                 buffer.offset() as vk::DeviceSize,
                                 buffer.size() as vk::DeviceSize, data);
            }

            Ok(self)
        }
    }

    /// Adds a command that writes the content of a buffer.
    ///
    /// The size of the buffer slice is what is taken into account when determining the size. If
    /// `data` is larger than the slice, then the remaining will be ignored. If `data` is smaller,
    /// an error is returned.
    ///
    /// If the size of the slice is 0, no command is added.
    ///
    /// # Panic
    ///
    /// - Panicks if the buffer was not allocated with the same device as this command buffer.
    ///
    pub fn update_buffer_untyped<'a, S, T: ?Sized, B, D: ?Sized>(mut self, buffer: S, data: &D)
                                                                 -> Result<UnsafeCommandBufferBuilder,
                                                                           BufferUpdateError>
        where S: Into<BufferSlice<'a, T, B>>,
              B: Buffer + Send + Sync + 'static
    {
        unsafe {
            let buffer = buffer.into();

            // Performing checks.
            if self.within_render_pass { return Err(BufferUpdateError::ForbiddenWithinRenderPass); }
            assert_eq!(buffer.buffer().inner_buffer().device().internal_object(),
                       self.pool.device().internal_object());
            if !buffer.buffer().inner_buffer().usage_transfer_src() {
                return Err(BufferUpdateError::WrongUsageFlag);
            }
            if (buffer.offset() % 4) != 0 || (buffer.size() % 4) != 0 {
                return Err(BufferUpdateError::WrongAlignment);
            }
            if buffer.size() == 0 { return Ok(self); }
            if buffer.size() >= 0x10000 { return Err(BufferUpdateError::RegionTooLarge); }
            if mem::size_of_val(data) < buffer.size() {
                return Err(BufferUpdateError::DataTooSmall);
            }

            // Adding the command.
            self.keep_alive.push(buffer.buffer().clone());

            {
                let vk = self.device.pointers();
                let cmd = self.cmd.clone().unwrap();
                vk.CmdUpdateBuffer(cmd, buffer.buffer().inner_buffer().internal_object(),
                                   buffer.offset() as vk::DeviceSize,
                                   (buffer.size() / 4) as vk::DeviceSize,
                                   data as *const D as *const u32);
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

error_ty!{BufferFillError => "Error that can happen when filling a buffer.",
    ForbiddenWithinRenderPass => "can't copy buffers from within a render pass",
    NotSupportedByQueueFamily => "the queue family this command buffer belongs to does not \
                                  support this operation",
    WrongUsageFlag => "one of the buffers doesn't have the correct usage flag",
    WrongAlignment => "the offset and size must be multiples of 4",
}

error_ty!{BufferUpdateError => "Error that can happen when updating a buffer.",
    ForbiddenWithinRenderPass => "can't copy buffers from within a render pass",
    WrongUsageFlag => "one of the buffers doesn't have the correct usage flag",
    WrongAlignment => "the offset and size must be multiples of 4",
    RegionTooLarge => "the size of the region exceeds the allowed limits",
    DataTooSmall => "the data is smaller than the region to copy to",
}
