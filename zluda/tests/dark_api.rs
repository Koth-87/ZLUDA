use crate::common::CudaDriverFns;
use cuda_types::*;
use std::{mem, ptr};
mod common;

cuda_driver_test!(heap_alloc_chain);

unsafe fn heap_alloc_chain<T: CudaDriverFns>(cuda: T) {
    assert_eq!(cuda.cuInit(0), CUresult::CUDA_SUCCESS);
    let mut export_table = mem::zeroed();
    let guid = zluda_dark_api::HeapAccess::GUID;
    assert_eq!(
        cuda.cuGetExportTable(&mut export_table, &guid),
        CUresult::CUDA_SUCCESS
    );
    assert_eq!(3 * mem::size_of::<usize>(), *export_table.cast::<usize>());
    let heap_access = zluda_dark_api::HeapAccess::new(export_table);
    let mut record1 = ptr::null_mut();
    assert_eq!(
        CUresult::CUDA_SUCCESS,
        heap_access.heap_alloc(&mut record1, None, 1)
    );
    let mut record2 = ptr::null_mut();
    assert_eq!(
        CUresult::CUDA_SUCCESS,
        heap_access.heap_alloc(&mut record2, None, 2)
    );
    let mut record3 = ptr::null_mut();
    assert_eq!(
        CUresult::CUDA_SUCCESS,
        heap_access.heap_alloc(&mut record3, None, 3)
    );
    assert_eq!((&*record1).destructor, None);
    assert_eq!((&*record1).value, 1);
    assert_eq!((&*record1).prev_alloc, record2);
    assert_eq!((&*record1).next_alloc, ptr::null());
    assert_eq!((&*record2).destructor, None);
    assert_eq!((&*record2).value, 2);
    assert_eq!((&*record2).prev_alloc, record3);
    assert_eq!((&*record2).next_alloc, record1);
    assert_eq!((&*record3).destructor, None);
    assert_eq!((&*record3).value, 3);
    assert_eq!((&*record3).prev_alloc, ptr::null());
    assert_eq!((&*record3).next_alloc, record2);
}

cuda_driver_test!(heap_free);

unsafe fn heap_free<T: CudaDriverFns>(cuda: T) {
    assert_eq!(cuda.cuInit(0), CUresult::CUDA_SUCCESS);
    let mut export_table = mem::zeroed();
    let guid = zluda_dark_api::HeapAccess::GUID;
    assert_eq!(
        cuda.cuGetExportTable(&mut export_table, &guid),
        CUresult::CUDA_SUCCESS
    );
    let heap_access = zluda_dark_api::HeapAccess::new(export_table);
    let mut record1 = ptr::null_mut();
    assert_eq!(
        CUresult::CUDA_SUCCESS,
        heap_access.heap_alloc(&mut record1, Some(shutdown), 11)
    );
    let mut record2 = ptr::null_mut();
    assert_eq!(
        CUresult::CUDA_SUCCESS,
        heap_access.heap_alloc(&mut record2, Some(shutdown), 12)
    );
    let mut record3 = ptr::null_mut();
    assert_eq!(
        CUresult::CUDA_SUCCESS,
        heap_access.heap_alloc(&mut record3, None, 13)
    );
    let mut value = 0usize;
    assert_eq!(
        CUresult::CUDA_SUCCESS,
        heap_access.heap_free(record2, &mut value),
    );
    assert_eq!(value, 12);
    assert_eq!(
        mem::transmute::<_, usize>((&*record1).destructor),
        shutdown as usize
    );
    assert_eq!((&*record1).value, 11);
    assert_eq!((&*record1).prev_alloc, record3);
    assert_eq!((&*record1).next_alloc, ptr::null());
    assert_eq!((&*record3).destructor, None);
    assert_eq!((&*record3).value, 13);
    assert_eq!((&*record3).prev_alloc, ptr::null());
    assert_eq!((&*record3).next_alloc, record1);
}

unsafe extern "system" fn shutdown(_unknown: u32, _value: usize) {}
