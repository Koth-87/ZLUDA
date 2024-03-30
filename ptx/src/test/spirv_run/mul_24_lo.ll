target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7"
target triple = "amdgcn-amd-amdhsa"

define protected amdgpu_kernel void @mul_24_lo(ptr addrspace(4) byref(i64) %"32", ptr addrspace(4) byref(i64) %"33") #0 {
"44":
  %"10" = alloca i1, align 1, addrspace(5)
  store i1 false, ptr addrspace(5) %"10", align 1
  %"11" = alloca i1, align 1, addrspace(5)
  store i1 false, ptr addrspace(5) %"11", align 1
  %"4" = alloca i64, align 8, addrspace(5)
  %"5" = alloca i64, align 8, addrspace(5)
  %"6" = alloca i32, align 4, addrspace(5)
  %"7" = alloca i32, align 4, addrspace(5)
  %"8" = alloca i32, align 4, addrspace(5)
  %"9" = alloca i32, align 4, addrspace(5)
  %"12" = load i64, ptr addrspace(4) %"32", align 8
  store i64 %"12", ptr addrspace(5) %"4", align 8
  %"13" = load i64, ptr addrspace(4) %"33", align 8
  store i64 %"13", ptr addrspace(5) %"5", align 8
  %"15" = load i64, ptr addrspace(5) %"4", align 8
  %"34" = inttoptr i64 %"15" to ptr
  %"14" = load i32, ptr %"34", align 4
  store i32 %"14", ptr addrspace(5) %"6", align 4
  %"17" = load i64, ptr addrspace(5) %"4", align 8
  %"35" = inttoptr i64 %"17" to ptr
  %"46" = getelementptr inbounds i8, ptr %"35", i64 4
  %"16" = load i32, ptr %"46", align 4
  store i32 %"16", ptr addrspace(5) %"7", align 4
  %"19" = load i32, ptr addrspace(5) %"6", align 4
  %"20" = load i32, ptr addrspace(5) %"7", align 4
  %0 = shl i32 %"19", 8
  %1 = ashr i32 %0, 8
  %2 = shl i32 %"20", 8
  %3 = ashr i32 %2, 8
  %"36" = mul i32 %1, %3
  store i32 %"36", ptr addrspace(5) %"8", align 4
  %"22" = load i32, ptr addrspace(5) %"6", align 4
  %"23" = load i32, ptr addrspace(5) %"7", align 4
  %4 = shl i32 %"22", 8
  %5 = lshr i32 %4, 8
  %6 = shl i32 %"23", 8
  %7 = lshr i32 %6, 8
  %"39" = mul i32 %5, %7
  store i32 %"39", ptr addrspace(5) %"9", align 4
  %"24" = load i64, ptr addrspace(5) %"5", align 8
  %"25" = load i32, ptr addrspace(5) %"8", align 4
  %"42" = inttoptr i64 %"24" to ptr
  store i32 %"25", ptr %"42", align 4
  %"26" = load i64, ptr addrspace(5) %"5", align 8
  %"27" = load i32, ptr addrspace(5) %"9", align 4
  %"43" = inttoptr i64 %"26" to ptr
  %"48" = getelementptr inbounds i8, ptr %"43", i64 4
  store i32 %"27", ptr %"48", align 4
  ret void
}

attributes #0 = { "amdgpu-unsafe-fp-atomics"="true" "denormal-fp-math"="ieee,ieee" "denormal-fp-math-f32"="ieee,ieee" "no-trapping-math"="true" "uniform-work-group-size"="true" }
