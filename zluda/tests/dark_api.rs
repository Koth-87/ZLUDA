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
    assert_eq!((&*record1).next_alloc, ptr::null_mut());
    assert_eq!((&*record2).destructor, None);
    assert_eq!((&*record2).value, 2);
    assert_eq!((&*record2).prev_alloc, record3);
    assert_eq!((&*record2).next_alloc, record1);
    assert_eq!((&*record3).destructor, None);
    assert_eq!((&*record3).value, 3);
    assert_eq!((&*record3).prev_alloc, ptr::null_mut());
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
    assert_eq!((&*record1).next_alloc, ptr::null_mut());
    assert_eq!((&*record3).destructor, None);
    assert_eq!((&*record3).value, 13);
    assert_eq!((&*record3).prev_alloc, ptr::null_mut());
    assert_eq!((&*record3).next_alloc, record1);
}

unsafe extern "system" fn shutdown(_unknown: u32, _value: usize) {}

cuda_driver_test!(dark_api_primary_context_allocate);

unsafe fn dark_api_primary_context_allocate<T: CudaDriverFns>(cuda: T) {
    assert_eq!(cuda.cuInit(0), CUresult::CUDA_SUCCESS);
    let dev = CUdevice_v1(0);
    let mut ctx1 = mem::zeroed();
    let mut export_table = mem::zeroed();
    let guid = zluda_dark_api::CudartInterface::GUID;
    assert_eq!(
        cuda.cuGetExportTable(&mut export_table, &guid),
        CUresult::CUDA_SUCCESS
    );
    let cudart_interface = zluda_dark_api::CudartInterface::new(export_table);
    assert_eq!(
        cudart_interface.primary_context_allocate(&mut ctx1, dev),
        CUresult::CUDA_SUCCESS
    );
    let mut api_version = mem::zeroed();
    assert_eq!(
        cuda.cuCtxGetApiVersion(ctx1, &mut api_version),
        CUresult::CUDA_ERROR_INVALID_CONTEXT
    );
    let mut flags = 0;
    let mut active = 0;
    assert_eq!(
        cuda.cuDevicePrimaryCtxGetState(dev, &mut flags, &mut active),
        CUresult::CUDA_SUCCESS
    );
    assert_eq!((flags, active), (0, 0));
    assert_eq!(cuda.cuCtxSetCurrent(ctx1), CUresult::CUDA_SUCCESS);
    assert_ne!(
        cuda.cuMemAlloc_v2(&mut mem::zeroed(), 4),
        CUresult::CUDA_SUCCESS
    );
    let mut device = mem::zeroed();
    assert_eq!(cuda.cuCtxGetDevice(&mut device), CUresult::CUDA_SUCCESS);
    // TODO: re-enable when adding context getters
    /*
    let mut cache_cfg = mem::zeroed();
    assert_eq!(
        cuda.cuCtxGetCacheConfig(&mut cache_cfg),
        CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED
    );
    let mut exec_affinity = mem::zeroed();
    assert_eq!(
        cuda.cuCtxGetExecAffinity(
            &mut exec_affinity,
            CUexecAffinityType::CU_EXEC_AFFINITY_TYPE_SM_COUNT
        ),
        CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED
    );
    let mut flags = mem::zeroed();
    assert_eq!(cuda.cuCtxGetFlags(&mut flags,), CUresult::CUDA_SUCCESS);
    let mut stack = mem::zeroed();
    assert_eq!(
        cuda.cuCtxGetLimit(&mut stack, CUlimit::CU_LIMIT_STACK_SIZE),
        CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED
    );
    let mut shared_mem_cfg = mem::zeroed();
    assert_eq!(
        cuda.cuCtxGetSharedMemConfig(&mut shared_mem_cfg),
        CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED
    );
    let mut lowest_priority = mem::zeroed();
    let mut highest_priority = mem::zeroed();
    assert_eq!(
        cuda.cuCtxGetStreamPriorityRange(&mut lowest_priority, &mut highest_priority),
        CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED
    );
     */
    let mut ctx2 = mem::zeroed();
    assert_eq!(
        cuda.cuDevicePrimaryCtxRetain(&mut ctx2, dev),
        CUresult::CUDA_SUCCESS
    );
    assert_eq!(ctx1, ctx2);
    assert_eq!(
        cuda.cuCtxGetApiVersion(ctx1, &mut api_version),
        CUresult::CUDA_SUCCESS
    );
    assert_eq!(cuda.cuCtxGetDevice(&mut device), CUresult::CUDA_SUCCESS);
    assert_eq!(
        cuda.cuDevicePrimaryCtxRelease_v2(dev),
        CUresult::CUDA_SUCCESS
    );
    assert_ne!(
        cuda.cuDevicePrimaryCtxRelease_v2(dev),
        CUresult::CUDA_SUCCESS
    );
}

cuda_driver_test!(dark_api_primary_context_create_with_flags);

unsafe fn dark_api_primary_context_create_with_flags<T: CudaDriverFns>(cuda: T) {
    assert_eq!(cuda.cuInit(0), CUresult::CUDA_SUCCESS);
    let dev = CUdevice_v1(0);
    let mut export_table = mem::zeroed();
    let guid = zluda_dark_api::CudartInterface::GUID;
    assert_eq!(
        cuda.cuGetExportTable(&mut export_table, &guid),
        CUresult::CUDA_SUCCESS
    );
    let cudart_interface = zluda_dark_api::CudartInterface::new(export_table);
    assert_eq!(
        cudart_interface.primary_context_create_with_flags(dev, 1),
        CUresult::CUDA_SUCCESS
    );
    let mut flags = 0;
    let mut active = 0;
    assert_eq!(
        cuda.cuDevicePrimaryCtxGetState(dev, &mut flags, &mut active),
        CUresult::CUDA_SUCCESS
    );
    assert_eq!((flags, active), (1, 1));
}

cuda_driver_test!(dark_api_primary_context_create_with_flags_fail);

unsafe fn dark_api_primary_context_create_with_flags_fail<T: CudaDriverFns>(cuda: T) {
    assert_eq!(cuda.cuInit(0), CUresult::CUDA_SUCCESS);
    let dev = CUdevice_v1(0);
    let mut export_table = mem::zeroed();
    let guid = zluda_dark_api::CudartInterface::GUID;
    assert_eq!(
        cuda.cuGetExportTable(&mut export_table, &guid),
        CUresult::CUDA_SUCCESS
    );
    let cudart_interface = zluda_dark_api::CudartInterface::new(export_table);
    assert_eq!(
        cuda.cuDevicePrimaryCtxRetain(&mut mem::zeroed(), dev),
        CUresult::CUDA_SUCCESS
    );
    assert_ne!(
        cudart_interface.primary_context_create_with_flags(dev, 1),
        CUresult::CUDA_SUCCESS
    );
}
