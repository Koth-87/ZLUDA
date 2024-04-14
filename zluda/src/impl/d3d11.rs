use cl3::d3d11::{
    CL_CONTEXT_D3D11_DEVICE_KHR, CL_D3D11_DXGI_ADAPTER_KHR, CL_PREFERRED_DEVICES_FOR_D3D11_KHR,
};
use cl3::device::*;
use cl3::ext::{CL_CONTEXT_PLATFORM, CL_DEVICE_TYPE_GPU, CL_MEM_READ_WRITE};
use cl3::platform::*;
use cl3::{ext::CL_PLATFORM_VENDOR, platform::get_platform_info};
use cuda_types::*;
use hip_runtime_sys::{hipDevice_t, hipGraphicsResource_t};
use lazy_static::lazy_static;
use rustc_hash::FxHashMap;
use std::ffi::{c_void, CStr};
use std::mem::ManuallyDrop;
use std::sync::Mutex;
use std::{mem, ptr};
use windows::Win32::Graphics::Direct3D11::{
    ID3D11Buffer, ID3D11Device, ID3D11Resource, ID3D11Texture2D, ID3D11Texture3D,
};
use windows::Win32::Graphics::Dxgi::IDXGIAdapter;

static mut PLATFORM: *mut c_void = ptr::null_mut();
static mut DEVICE: *mut c_void = ptr::null_mut();

pub(crate) unsafe fn get_device(
    p_cuda_device: *mut hipDevice_t,
    p_adapter: *mut *mut std::ffi::c_void,
) -> Result<(), cuda_types::CUresult> {
    initialize_opencl();
    let clGetDeviceIDsFromD3D11KHR =
        mem::transmute::<_, cl3::d3d11::clGetDeviceIDsFromD3D11KHR_fn>(
            cl3::ext::clGetExtensionFunctionAddressForPlatform(
                PLATFORM,
                b"clGetDeviceIDsFromD3D11KHR\0".as_ptr().cast(),
            ),
        )
        .unwrap();
    let mut dev = mem::zeroed();
    let mut dev_count = mem::zeroed();
    let p_adapter = p_adapter.cast();
    let p_adapter = IDXGIAdapter::from_raw_borrowed(&p_adapter).unwrap();
    assert_eq!(
        0,
        clGetDeviceIDsFromD3D11KHR(
            PLATFORM,
            CL_D3D11_DXGI_ADAPTER_KHR,
            p_adapter.as_raw(),
            CL_PREFERRED_DEVICES_FOR_D3D11_KHR,
            1,
            &mut dev,
            &mut dev_count
        )
    );
    assert_eq!(dev_count, 1);
    DEVICE = dev;
    *p_cuda_device = 0;
    Ok(())
}

use windows::core::Interface;

pub(crate) unsafe fn register_resource(
    p_cuda_device: *mut hipGraphicsResource_t,
    p_d3_dresource: *const std::ffi::c_void,
    flags: u32,
) -> Result<(), cuda_types::CUresult> {
    if flags != 0 {
        panic!()
    }
    let p_d3_dresource = p_d3_dresource.cast_mut();
    let resource = ID3D11Resource::from_raw_borrowed(&p_d3_dresource).unwrap();
    let mem = if let Ok(buffer) = resource.cast::<ID3D11Buffer>() {
        let device = buffer.GetDevice().unwrap();
        let clCreateFromD3D11BufferKHR =
            mem::transmute::<_, cl3::d3d11::clCreateFromD3D11BufferKHR_fn>(
                cl3::ext::clGetExtensionFunctionAddressForPlatform(
                    PLATFORM,
                    b"clCreateFromD3D11BufferKHR\0".as_ptr().cast(),
                ),
            )
            .unwrap();
        let mut err = 0;
        let mem = clCreateFromD3D11BufferKHR(
            get_cl_context(device),
            CL_MEM_READ_WRITE,
            buffer.as_raw(),
            &mut err,
        );
        assert_eq!(err, 0);
        mem
    } else if let Ok(tex_2d) = resource.cast::<ID3D11Texture2D>() {
        let device = tex_2d.GetDevice().unwrap();
        let clCreateFromD3D11Texture2DKHR =
            mem::transmute::<_, cl3::d3d11::clCreateFromD3D11Texture2DKHR_fn>(
                cl3::ext::clGetExtensionFunctionAddressForPlatform(
                    PLATFORM,
                    b"clCreateFromD3D11Texture2DKHR\0".as_ptr().cast(),
                ),
            )
            .unwrap();
        let mut err = 0;
        let mem = clCreateFromD3D11Texture2DKHR(
            get_cl_context(device),
            CL_MEM_READ_WRITE,
            tex_2d.as_raw(),
            0,
            &mut err,
        );
        assert_eq!(err, 0);
        mem
    } else if let Ok(tex_3d) = resource.cast::<ID3D11Texture3D>() {
        let device = tex_3d.GetDevice().unwrap();
        let clCreateFromD3D11Texture3DKHR =
            mem::transmute::<_, cl3::d3d11::clCreateFromD3D11Texture3DKHR_fn>(
                cl3::ext::clGetExtensionFunctionAddressForPlatform(
                    PLATFORM,
                    b"clCreateFromD3D11Texture3DKHR\0".as_ptr().cast(),
                ),
            )
            .unwrap();
        let mut err = 0;
        let mem = clCreateFromD3D11Texture3DKHR(
            get_cl_context(device),
            CL_MEM_READ_WRITE,
            tex_3d.as_raw(),
            0,
            &mut err,
        );
        assert_eq!(err, 0);
        mem
    } else {
        panic!("")
    };
    *p_cuda_device = mem.cast();
    Ok(())
}

unsafe fn initialize_opencl() {
    let platform_ids = cl3::platform::get_platform_ids().unwrap();
    for platform_id in platform_ids {
        let devices = get_device_ids(platform_id, CL_DEVICE_TYPE_GPU).unwrap();
        for dev in devices {
            let vendor_id = get_device_info(dev, CL_DEVICE_VENDOR_ID).unwrap().to_uint();
            dbg!(vendor_id);
            if vendor_id != 0x1002 {
                continue;
            }
            let name = get_device_info(dev, CL_DEVICE_NAME).unwrap().to_vec_uchar();
            let name = String::from_utf8(name).unwrap();
            dbg!(&name);
            if name != "gfx1030\0" {
                continue;
            }
            unsafe { PLATFORM = platform_id };
            //unsafe { DEVICE = dev };
            return;
        }
    }
    panic!()
}

lazy_static! {
    static ref DEVICE_TO_CONTEXT: Mutex<FxHashMap<usize, usize>> = Mutex::new(FxHashMap::default());
}

unsafe fn get_cl_context(dev: ID3D11Device) -> *mut c_void {
    let mut map = DEVICE_TO_CONTEXT.lock().unwrap();
    match map.entry(dev.as_raw() as usize) {
        std::collections::hash_map::Entry::Occupied(entry) => (*entry.get()) as *mut c_void,
        std::collections::hash_map::Entry::Vacant(entry) => {
            let properties = [
                CL_CONTEXT_PLATFORM,
                PLATFORM as _,
                CL_CONTEXT_D3D11_DEVICE_KHR as _,
                dev.as_raw() as _,
                0,
            ];
            let context =
                cl3::context::create_context(&[DEVICE], properties.as_ptr(), None, ptr::null_mut())
                    .unwrap();
            entry.insert(context as usize);
            context
        }
    }
}
