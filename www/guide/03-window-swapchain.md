---
layout: page
title: "Tutorial 3: windows and swapchains"
---

# Windows and swapchains

Vulkan can be used to perform calculations (like OpenCL for example), but its main usage is to
draw graphics. And before we can draw graphics, we have to create a window where to display
the result.

## Creating a window

Creating a window is out of the scope of Vulkan. Instead we have to use platform-specific
functionnalities dedicated to opening a window.

For the purpose of this tutorial, we are going to use the `winit` and the `vulkano-win` crates.
The former will be used to open a window and handle keyboard and mouse input, and the latter
is used as a glue between `winit` and `vulkano`.

To do so, let's add these dependencies to our Cargo.toml:

{% highlight toml %}
winit = "0.5"
vulkano-win = "0.1"
{% endhighlight %}

... and to our Rust code:

{% highlight rust %}
extern crate winit;
extern crate vulkano_win;
{% endhighlight %}

Creating a window is as easy as this:

{% highlight rust %}
use vulkano_win::VkSurfaceBuild;

let window = winit::WindowBuilder::new().build_vk_surface(&instance).unwrap();
{% endhighlight %}

This code creates a window with the default parameters, and also builds a Vulkan *surface* object
that represents the surface of that window whenever the Vulkan API is concerned.
Calling `window.window()` will return an object that allows you to manipulate the window, and
calling `window.surface()` will return a `Surface` object of `vulkano`.

However, if you try to run this code you will notice that the `build_vk_surface` returns an error.
The reason is that surfaces are actually not part of Vulkan itself, but of several *extension*s
to the Vulkan API. These extensions are disabled by default and need to be manually enabled when
creating the instance before one can use their capabilities.

Fortunately the `vulkano_win` provides a function named `required_extensions()` that will return
a list of the extensions that are needed.

In order to make this work, we need to modify the way the instance is created:

{% highlight rust %}
let instance = {
    let extensions = vulkano_win::required_extensions();
    Instance::new(None, &extensions, None).expect("failed to create Vulkan instance")
};
{% endhighlight %}

After you made the change, running the program should now work and open then immediately close
a window.

## Events handling

## Creating a swapchain
