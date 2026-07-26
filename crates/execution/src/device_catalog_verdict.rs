//! Small terminal GPU operators for catalog-only authorization and type presentation.
//!
//! These operators deliberately accept complete, unfiltered catalog candidates and a typed request.
//! They are bounded control-plane work, not a replacement for resident catalog relations.  The
//! host marshals immutable facts and decodes one terminal device verdict; it never derives an ACL
//! allow/deny or a `format_type` display string.

use crate::{check_cuda, launch_on_pooled_stream, CudaResidentDeviceMemory, CudaRuntimeProbeError};
use std::ffi::c_void;

const MAX_FORMAT_TYPE_TEXT: usize = 256;
/// The `psql \\df` terminal catalog operator keeps each displayed catalog cell bounded.  The
/// PostgreSQL catalog identifiers this route models are already bounded well below this limit;
/// an overlong candidate is a device-stage error rather than a host formatting escape hatch.
pub const MAX_FUNCTION_LIST_TEXT: usize = 256;
pub const FUNCTION_LIST_COLUMN_COUNT: usize = 13;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceAclRole {
    pub oid: u32,
    pub name_offset: u32,
    pub name_len: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceAclObject {
    pub oid: u32,
    pub kind: u32,
    pub name_offset: u32,
    pub name_len: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceAclGrant {
    pub object_oid: u32,
    pub object_kind: u32,
    pub grantee_name_offset: u32,
    pub grantee_name_len: u32,
    pub privilege: u32,
    /// Raw catalog identity for PostgreSQL's PUBLIC pseudo-role.  This is not an allow bit;
    /// the terminal kernel still matches the object and requested privilege.
    pub grantee_is_public: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceAclSchemaGrant {
    pub grantee_name_offset: u32,
    pub grantee_name_len: u32,
    pub privilege: u32,
    /// Raw catalog identity for PostgreSQL's PUBLIC pseudo-role.
    pub grantee_is_public: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceAclRequest {
    pub principal_oid: u32,
    pub bootstrap: u32,
    pub target_kind: u32,
    pub privilege: u32,
    pub target_name_offset: u32,
    pub target_name_len: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceAclVerdict {
    Allowed,
    /// The role/object identity or target-object privilege did not authorize the request.
    DeniedObject,
    /// The request reached a user object but lacks the required public-schema USAGE/CREATE grant.
    DeniedSchema,
    MissingObject,
    InvalidInput,
}

/// A role-name request whose text lives in the same complete catalog byte relation as the roles.
/// Keeping the request typed prevents the control plane from deriving an OID from a host map.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceRoleNameRequest {
    pub name_offset: u32,
    pub name_len: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceRoleNameVerdict {
    Found(u32),
    Missing,
    InvalidInput,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceFormatTypeCandidate {
    pub oid: i32,
    pub display_offset: u32,
    pub display_len: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceFormatTypeValue {
    pub ordinal: u32,
    pub name_offset: u32,
    pub name_len: u32,
    pub type_oid: i32,
    pub typmod: i32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceFormatTypeRow {
    pub ordinal: u32,
    pub name_len: u32,
    pub type_len: u32,
    pub name: [u8; MAX_FORMAT_TYPE_TEXT],
    pub display: [u8; MAX_FORMAT_TYPE_TEXT],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceFormatTypeVerdict {
    Complete,
    UnknownOid,
    InvalidTypmod,
    InvalidInput,
}

/// Raw `pg_proc` facts consumed by the terminal `psql \\df` catalog operator.  Presentation is
/// intentionally not encoded here: namespace/type/owner/language/comment and ACL candidates are
/// resolved on the device.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceFunctionListFunction {
    pub oid: u32,
    pub namespace_oid: u32,
    pub owner_oid: u32,
    pub return_type_oid: i32,
    pub language_oid: u32,
    pub name_offset: u32,
    pub name_len: u32,
    pub prokind: u32,
    pub provolatile: u32,
    pub proparallel: u32,
    pub prosecdef: u32,
}

/// A typed catalog OID-to-text candidate.  The same wire record represents namespaces, result
/// types, role names, and languages; each relation remains a separate complete input to the
/// device operator.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceFunctionListTextCandidate {
    pub oid: u32,
    pub text_offset: u32,
    pub text_len: u32,
}

/// Raw `pg_description` facts.  The terminal operator retains the class/sub-id predicates so
/// host staging cannot select a function comment by itself.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceFunctionListDescription {
    pub class_oid: i32,
    pub object_oid: u32,
    pub object_sub_id: i32,
    pub text_offset: u32,
    pub text_len: u32,
}

/// Complete raw function-grant candidate.  `privilege` is deliberately not pre-filtered by the
/// host; the terminal program applies the EXECUTE presentation predicate and formats the result.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceFunctionListGrant {
    pub function_oid: u32,
    pub privilege: u32,
    pub grantee_offset: u32,
    pub grantee_len: u32,
    pub grantee_is_public: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceFunctionListRequest {
    pub name_filter_offset: u32,
    pub name_filter_len: u32,
    pub verbose: u32,
}

/// Every cell is materialized by the terminal GPU operator.  `u32::MAX` is SQL NULL; otherwise
/// the length indexes `bytes`.  Keeping the row fixed-width makes the sole D2H result
/// transfer auditable and avoids reconstructing presentation rows on the host.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceFunctionListCell {
    pub len: u32,
    pub bytes: [u8; MAX_FUNCTION_LIST_TEXT],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceFunctionListRow {
    pub ordinal: u32,
    pub function_oid: u32,
    pub cells: [DeviceFunctionListCell; FUNCTION_LIST_COLUMN_COUNT],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceFunctionListVerdict {
    Complete,
    InvalidInput,
}

type CuMemcpyHtoD = unsafe extern "C" fn(u64, *const c_void, usize) -> i32;
type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
type CuLaunchKernel = unsafe extern "C" fn(
    *mut c_void,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    *mut c_void,
    *mut *mut c_void,
    *mut *mut c_void,
) -> i32;

impl CudaResidentDeviceMemory {
    /// Resolve the complete ACL candidate set and return a single GPU-computed verdict.
    #[allow(clippy::too_many_arguments)]
    pub fn catalog_acl_verdict(
        &self,
        request: DeviceAclRequest,
        roles: &[DeviceAclRole],
        objects: &[DeviceAclObject],
        grants: &[DeviceAclGrant],
        schema_grants: &[DeviceAclSchemaGrant],
        bytes: &[u8],
    ) -> Result<DeviceAclVerdict, CudaRuntimeProbeError> {
        let primary = self.primary();
        primary.set_current()?;
        let roles_bytes = bytes_for(roles)?;
        let objects_bytes = bytes_for(objects)?;
        let grants_bytes = bytes_for(grants)?;
        let schema_grants_bytes = bytes_for(schema_grants)?;
        let bytes_len = bytes.len();
        let request_dev = upload(primary, self, std::slice::from_ref(&request))?;
        let roles_dev = upload(primary, self, roles)?;
        let objects_dev = upload(primary, self, objects)?;
        let grants_dev = upload(primary, self, grants)?;
        let schema_grants_dev = upload(primary, self, schema_grants)?;
        let bytes_dev = upload_bytes(primary, self, bytes)?;
        let mut ptx = ACL_PTX.to_vec();
        ptx.push(0);
        let function = primary.cached_function(c"gpu_db_catalog_acl_verdict", &ptx)?;
        let launch = cuda_launch(self)?;
        let mut output = [0_u8; 4];
        launch_on_pooled_stream(self, Some(&mut output), |stream, output_ptr| {
            let mut request_arg = request_dev.ptr;
            let mut roles_arg = roles_dev.ptr;
            let mut roles_count = roles.len() as u64;
            let mut objects_arg = objects_dev.ptr;
            let mut objects_count = objects.len() as u64;
            let mut grants_arg = grants_dev.ptr;
            let mut grants_count = grants.len() as u64;
            let mut schema_grants_arg = schema_grants_dev.ptr;
            let mut schema_grants_count = schema_grants.len() as u64;
            let mut bytes_arg = bytes_dev.ptr;
            let mut bytes_count = bytes_len as u64;
            let mut output_arg = output_ptr;
            let mut args = [
                (&mut request_arg as *mut u64).cast::<c_void>(),
                (&mut roles_arg as *mut u64).cast::<c_void>(),
                (&mut roles_count as *mut u64).cast::<c_void>(),
                (&mut objects_arg as *mut u64).cast::<c_void>(),
                (&mut objects_count as *mut u64).cast::<c_void>(),
                (&mut grants_arg as *mut u64).cast::<c_void>(),
                (&mut grants_count as *mut u64).cast::<c_void>(),
                (&mut schema_grants_arg as *mut u64).cast::<c_void>(),
                (&mut schema_grants_count as *mut u64).cast::<c_void>(),
                (&mut bytes_arg as *mut u64).cast::<c_void>(),
                (&mut bytes_count as *mut u64).cast::<c_void>(),
                (&mut output_arg as *mut u64).cast::<c_void>(),
            ];
            unsafe {
                launch(
                    function,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0,
                    stream,
                    args.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            }
        })?;
        // Keep byte-count computations adjacent to the upload so changing one wire record cannot
        // accidentally leave a stale launch shape.  The variables are intentionally consumed here.
        let _ = (
            roles_bytes,
            objects_bytes,
            grants_bytes,
            schema_grants_bytes,
        );
        Ok(match u32::from_le_bytes(output) {
            1 => DeviceAclVerdict::Allowed,
            2 => DeviceAclVerdict::DeniedSchema,
            3 => DeviceAclVerdict::MissingObject,
            5 => DeviceAclVerdict::DeniedObject,
            _ => DeviceAclVerdict::InvalidInput,
        })
    }

    /// Resolve one SQL-visible role name against the complete device role relation.
    ///
    /// A duplicate candidate is malformed rather than being resolved by host ordering, and a
    /// missing candidate remains distinct from a device/input failure.
    pub fn catalog_role_name_verdict(
        &self,
        request: DeviceRoleNameRequest,
        roles: &[DeviceAclRole],
        bytes: &[u8],
    ) -> Result<DeviceRoleNameVerdict, CudaRuntimeProbeError> {
        let primary = self.primary();
        primary.set_current()?;
        let request_dev = upload(primary, self, std::slice::from_ref(&request))?;
        let roles_dev = upload(primary, self, roles)?;
        let bytes_dev = upload_bytes(primary, self, bytes)?;
        let mut ptx = ROLE_NAME_PTX.to_vec();
        ptx.push(0);
        let function = primary.cached_function(c"gpu_db_catalog_role_name_verdict", &ptx)?;
        let launch = cuda_launch(self)?;
        let mut output = [0_u8; 8];
        launch_on_pooled_stream(self, Some(&mut output), |stream, output_ptr| {
            let mut request_arg = request_dev.ptr;
            let mut roles_arg = roles_dev.ptr;
            let mut roles_count = roles.len() as u64;
            let mut bytes_arg = bytes_dev.ptr;
            let mut bytes_count = bytes.len() as u64;
            let mut output_arg = output_ptr;
            let mut args = [
                (&mut request_arg as *mut u64).cast::<c_void>(),
                (&mut roles_arg as *mut u64).cast::<c_void>(),
                (&mut roles_count as *mut u64).cast::<c_void>(),
                (&mut bytes_arg as *mut u64).cast::<c_void>(),
                (&mut bytes_count as *mut u64).cast::<c_void>(),
                (&mut output_arg as *mut u64).cast::<c_void>(),
            ];
            unsafe {
                launch(
                    function,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0,
                    stream,
                    args.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            }
        })?;
        let oid = u32::from_le_bytes(output[4..8].try_into().expect("fixed role verdict"));
        Ok(
            match u32::from_le_bytes(output[..4].try_into().expect("fixed role verdict")) {
                1 => DeviceRoleNameVerdict::Found(oid),
                2 => DeviceRoleNameVerdict::Missing,
                _ => DeviceRoleNameVerdict::InvalidInput,
            },
        )
    }

    /// Resolve OID/typmod pairs and materialize every final `format_type` cell on the device.
    pub fn catalog_format_types(
        &self,
        values: &[DeviceFormatTypeValue],
        candidates: &[DeviceFormatTypeCandidate],
        bytes: &[u8],
    ) -> Result<(DeviceFormatTypeVerdict, Vec<DeviceFormatTypeRow>), CudaRuntimeProbeError> {
        let primary = self.primary();
        primary.set_current()?;
        let output_bytes = values
            .len()
            .checked_mul(std::mem::size_of::<DeviceFormatTypeRow>())
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(values.len()))?;
        let values_dev = upload(primary, self, values)?;
        let candidates_dev = upload(primary, self, candidates)?;
        let bytes_dev = upload_bytes(primary, self, bytes)?;
        let output_dev = primary.lease_device_buffer(output_bytes.max(1))?;
        let mut ptx = FORMAT_TYPE_PTX.to_vec();
        ptx.push(0);
        let function = primary.cached_function(c"gpu_db_catalog_format_types", &ptx)?;
        let launch = cuda_launch(self)?;
        let mut verdict = [0_u8; 4];
        launch_on_pooled_stream(self, Some(&mut verdict), |stream, verdict_ptr| {
            let mut values_arg = values_dev.ptr;
            let mut value_count = values.len() as u64;
            let mut candidates_arg = candidates_dev.ptr;
            let mut candidate_count = candidates.len() as u64;
            let mut bytes_arg = bytes_dev.ptr;
            let mut bytes_count = bytes.len() as u64;
            let mut output_arg = output_dev.ptr;
            let mut verdict_arg = verdict_ptr;
            let mut args = [
                (&mut values_arg as *mut u64).cast::<c_void>(),
                (&mut value_count as *mut u64).cast::<c_void>(),
                (&mut candidates_arg as *mut u64).cast::<c_void>(),
                (&mut candidate_count as *mut u64).cast::<c_void>(),
                (&mut bytes_arg as *mut u64).cast::<c_void>(),
                (&mut bytes_count as *mut u64).cast::<c_void>(),
                (&mut output_arg as *mut u64).cast::<c_void>(),
                (&mut verdict_arg as *mut u64).cast::<c_void>(),
            ];
            unsafe {
                launch(
                    function,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0,
                    stream,
                    args.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            }
        })?;
        let verdict = match u32::from_le_bytes(verdict) {
            0 => DeviceFormatTypeVerdict::Complete,
            1 => DeviceFormatTypeVerdict::UnknownOid,
            2 => DeviceFormatTypeVerdict::InvalidTypmod,
            _ => DeviceFormatTypeVerdict::InvalidInput,
        };
        if !matches!(verdict, DeviceFormatTypeVerdict::Complete) {
            return Ok((verdict, Vec::new()));
        }
        let mut rows = vec![
            DeviceFormatTypeRow {
                ordinal: 0,
                name_len: 0,
                type_len: 0,
                name: [0; MAX_FORMAT_TYPE_TEXT],
                display: [0; MAX_FORMAT_TYPE_TEXT],
            };
            values.len()
        ];
        if output_bytes != 0 {
            let copy = cuda_dtoh(self)?;
            check_cuda(unsafe {
                copy(
                    rows.as_mut_ptr().cast::<c_void>(),
                    output_dev.ptr,
                    output_bytes,
                )
            })?;
        }
        Ok((verdict, rows))
    }

    /// Run the complete `psql \\df[+]` catalog program as one terminal device operator.  The
    /// caller supplies immutable raw catalog candidates from one snapshot; the device performs
    /// every OID join, name filter, sort, ACL formatting, and final projection before the single
    /// result frame is copied back.
    #[allow(clippy::too_many_arguments)]
    pub fn catalog_function_list(
        &self,
        request: DeviceFunctionListRequest,
        functions: &[DeviceFunctionListFunction],
        namespaces: &[DeviceFunctionListTextCandidate],
        types: &[DeviceFunctionListTextCandidate],
        owners: &[DeviceFunctionListTextCandidate],
        languages: &[DeviceFunctionListTextCandidate],
        descriptions: &[DeviceFunctionListDescription],
        grants: &[DeviceFunctionListGrant],
        bytes: &[u8],
    ) -> Result<(DeviceFunctionListVerdict, Vec<DeviceFunctionListRow>), CudaRuntimeProbeError>
    {
        let primary = self.primary();
        primary.set_current()?;
        let output_bytes = functions
            .len()
            .checked_mul(std::mem::size_of::<DeviceFunctionListRow>())
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(functions.len()))?;
        let request_dev = upload(primary, self, std::slice::from_ref(&request))?;
        let functions_dev = upload(primary, self, functions)?;
        let namespaces_dev = upload(primary, self, namespaces)?;
        let types_dev = upload(primary, self, types)?;
        let owners_dev = upload(primary, self, owners)?;
        let languages_dev = upload(primary, self, languages)?;
        let descriptions_dev = upload(primary, self, descriptions)?;
        let grants_dev = upload(primary, self, grants)?;
        let bytes_dev = upload_bytes(primary, self, bytes)?;
        let output_dev = primary.lease_device_buffer(output_bytes.max(1))?;
        let mut ptx = FUNCTION_LIST_PTX.to_vec();
        ptx.push(0);
        let function = primary.cached_function(c"gpu_db_catalog_function_list", &ptx)?;
        let launch = cuda_launch(self)?;
        // The launch helper owns the small device-to-host terminal header.  The much larger row
        // frame remains one fixed-width D2H gather below, never a host-assembled result.
        let mut header = [u8::MAX; 8];
        launch_on_pooled_stream(self, Some(&mut header), |stream, header_ptr| {
            let mut request_arg = request_dev.ptr;
            let mut functions_arg = functions_dev.ptr;
            let mut functions_count = functions.len() as u64;
            let mut namespaces_arg = namespaces_dev.ptr;
            let mut namespaces_count = namespaces.len() as u64;
            let mut types_arg = types_dev.ptr;
            let mut types_count = types.len() as u64;
            let mut owners_arg = owners_dev.ptr;
            let mut owners_count = owners.len() as u64;
            let mut languages_arg = languages_dev.ptr;
            let mut languages_count = languages.len() as u64;
            let mut descriptions_arg = descriptions_dev.ptr;
            let mut descriptions_count = descriptions.len() as u64;
            let mut grants_arg = grants_dev.ptr;
            let mut grants_count = grants.len() as u64;
            let mut bytes_arg = bytes_dev.ptr;
            let mut bytes_count = bytes.len() as u64;
            let mut output_arg = output_dev.ptr;
            let mut header_arg = header_ptr;
            let mut args = [
                (&mut request_arg as *mut u64).cast::<c_void>(),
                (&mut functions_arg as *mut u64).cast::<c_void>(),
                (&mut functions_count as *mut u64).cast::<c_void>(),
                (&mut namespaces_arg as *mut u64).cast::<c_void>(),
                (&mut namespaces_count as *mut u64).cast::<c_void>(),
                (&mut types_arg as *mut u64).cast::<c_void>(),
                (&mut types_count as *mut u64).cast::<c_void>(),
                (&mut owners_arg as *mut u64).cast::<c_void>(),
                (&mut owners_count as *mut u64).cast::<c_void>(),
                (&mut languages_arg as *mut u64).cast::<c_void>(),
                (&mut languages_count as *mut u64).cast::<c_void>(),
                (&mut descriptions_arg as *mut u64).cast::<c_void>(),
                (&mut descriptions_count as *mut u64).cast::<c_void>(),
                (&mut grants_arg as *mut u64).cast::<c_void>(),
                (&mut grants_count as *mut u64).cast::<c_void>(),
                (&mut bytes_arg as *mut u64).cast::<c_void>(),
                (&mut bytes_count as *mut u64).cast::<c_void>(),
                (&mut output_arg as *mut u64).cast::<c_void>(),
                (&mut header_arg as *mut u64).cast::<c_void>(),
            ];
            unsafe {
                launch(
                    function,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0,
                    stream,
                    args.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            }
        })?;
        let verdict = match u32::from_le_bytes(header[..4].try_into().expect("fixed header")) {
            0 => DeviceFunctionListVerdict::Complete,
            _ => DeviceFunctionListVerdict::InvalidInput,
        };
        let count = usize::try_from(u32::from_le_bytes(
            header[4..].try_into().expect("fixed header"),
        ))
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(functions.len()))?;
        if !matches!(verdict, DeviceFunctionListVerdict::Complete) {
            return Ok((verdict, Vec::new()));
        }
        if count > functions.len() {
            return Ok((DeviceFunctionListVerdict::InvalidInput, Vec::new()));
        }
        let mut rows = vec![
            DeviceFunctionListRow {
                ordinal: 0,
                function_oid: 0,
                cells: [DeviceFunctionListCell {
                    len: 0,
                    bytes: [0; MAX_FUNCTION_LIST_TEXT],
                }; FUNCTION_LIST_COLUMN_COUNT],
            };
            count
        ];
        if output_bytes != 0 && count != 0 {
            let copy = cuda_dtoh(self)?;
            check_cuda(unsafe {
                copy(
                    rows.as_mut_ptr().cast::<c_void>(),
                    output_dev.ptr,
                    count * std::mem::size_of::<DeviceFunctionListRow>(),
                )
            })?;
        }
        Ok((DeviceFunctionListVerdict::Complete, rows))
    }
}

fn bytes_for<T>(values: &[T]) -> Result<usize, CudaRuntimeProbeError> {
    values
        .len()
        .checked_mul(std::mem::size_of::<T>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(values.len()))
}

fn upload<'a, T>(
    primary: &'a crate::GpuPrimaryContext,
    resident: &CudaResidentDeviceMemory,
    values: &[T],
) -> Result<crate::PooledBufferLease<'a>, CudaRuntimeProbeError> {
    let byte_len = bytes_for(values)?;
    let buffer = primary.lease_device_buffer(byte_len.max(1))?;
    if byte_len != 0 {
        let copy = cuda_htod(resident)?;
        check_cuda(unsafe { copy(buffer.ptr, values.as_ptr().cast::<c_void>(), byte_len) })?;
    }
    Ok(buffer)
}

fn upload_bytes<'a>(
    primary: &'a crate::GpuPrimaryContext,
    resident: &CudaResidentDeviceMemory,
    values: &[u8],
) -> Result<crate::PooledBufferLease<'a>, CudaRuntimeProbeError> {
    let buffer = primary.lease_device_buffer(values.len().max(1))?;
    if !values.is_empty() {
        let copy = cuda_htod(resident)?;
        check_cuda(unsafe { copy(buffer.ptr, values.as_ptr().cast::<c_void>(), values.len()) })?;
    }
    Ok(buffer)
}

fn cuda_htod(resident: &CudaResidentDeviceMemory) -> Result<CuMemcpyHtoD, CudaRuntimeProbeError> {
    unsafe {
        resident
            .lib()
            .get::<CuMemcpyHtoD>(b"cuMemcpyHtoD_v2\0")
            .or_else(|_| resident.lib().get::<CuMemcpyHtoD>(b"cuMemcpyHtoD\0"))
            .map(|symbol| *symbol)
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)
    }
}

fn cuda_dtoh(resident: &CudaResidentDeviceMemory) -> Result<CuMemcpyDtoH, CudaRuntimeProbeError> {
    unsafe {
        resident
            .lib()
            .get::<CuMemcpyDtoH>(b"cuMemcpyDtoH_v2\0")
            .or_else(|_| resident.lib().get::<CuMemcpyDtoH>(b"cuMemcpyDtoH\0"))
            .map(|symbol| *symbol)
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)
    }
}

fn cuda_launch(
    resident: &CudaResidentDeviceMemory,
) -> Result<CuLaunchKernel, CudaRuntimeProbeError> {
    unsafe {
        resident
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map(|symbol| *symbol)
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)
    }
}

// Object: oid, kind, name offset, name len (all u32).  Grant: object oid, object kind,
// grantee name offset, grantee name len, privilege, public-identity.  The one-thread control operator is bounded
// by the catalog compatibility envelope and deliberately keeps its complete policy evaluation in
// one terminal device program.
const ACL_PTX: &[u8] = br#"
.version 6.0
.target sm_60
.address_size 64
.visible .entry gpu_db_catalog_acl_verdict(
 .param .u64 reqp, .param .u64 rolesp, .param .u64 rolesn,
 .param .u64 objp, .param .u64 objn, .param .u64 grantp, .param .u64 grantn,
 .param .u64 sgrantp, .param .u64 sgrantn, .param .u64 bytesp, .param .u64 bytesn,
 .param .u64 outp) {
 .reg .pred %p<18>; .reg .u32 %r<48>; .reg .u64 %q<48>;
 ld.param.u64 %q0,[reqp]; ld.param.u64 %q1,[rolesp]; ld.param.u64 %q2,[rolesn];
 ld.param.u64 %q3,[objp]; ld.param.u64 %q4,[objn]; ld.param.u64 %q5,[grantp]; ld.param.u64 %q6,[grantn];
 ld.param.u64 %q7,[sgrantp]; ld.param.u64 %q8,[sgrantn]; ld.param.u64 %q9,[bytesp]; ld.param.u64 %q10,[bytesn]; ld.param.u64 %q11,[outp];
 st.global.u32 [%q11],4;
 ld.global.u32 %r0,[%q0]; ld.global.u32 %r1,[%q0+4]; ld.global.u32 %r2,[%q0+8]; ld.global.u32 %r3,[%q0+12]; ld.global.u32 %r4,[%q0+16]; ld.global.u32 %r5,[%q0+20];
 add.u32 %r6,%r4,%r5; cvt.u64.u32 %q12,%r6; setp.gt.u64 %p0,%q12,%q10; @%p0 bra done;
 // Locate the exact target name + kind across complete object candidates.
 mov.u64 %q13,0; mov.u32 %r7,0; mov.u32 %r8,0;
obj_loop: setp.ge.u64 %p1,%q13,%q4; @%p1 bra obj_done;
 mul.lo.u64 %q14,%q13,16; add.u64 %q15,%q3,%q14; ld.global.u32 %r9,[%q15]; ld.global.u32 %r10,[%q15+4]; ld.global.u32 %r11,[%q15+8]; ld.global.u32 %r12,[%q15+12];
 // A system candidate is eligible only for a relation SELECT.  A same-named public base table
 // must retain its INSERT/UPDATE/DELETE route, while a bare SELECT remains ambiguously shadowed.
 setp.eq.u32 %p2,%r10,%r2; @%p2 bra obj_kind_ok; setp.ne.u32 %p2,%r10,5; @%p2 bra obj_next; setp.ne.u32 %p3,%r2,1; @%p3 bra obj_next; setp.ne.u32 %p3,%r3,1; @%p3 bra obj_next;
obj_kind_ok: setp.ne.u32 %p3,%r12,%r5; @%p3 bra obj_next;
 add.u32 %r13,%r11,%r12; cvt.u64.u32 %q16,%r13; setp.gt.u64 %p4,%q16,%q10; @%p4 bra done;
 mov.u32 %r14,0;
name_loop: setp.ge.u32 %p5,%r14,%r5; @%p5 bra name_equal; cvt.u64.u32 %q36,%r11; add.u64 %q17,%q9,%q36; cvt.u64.u32 %q37,%r14; add.u64 %q17,%q17,%q37; ld.global.u8 %r15,[%q17]; cvt.u64.u32 %q36,%r4; add.u64 %q18,%q9,%q36; add.u64 %q18,%q18,%q37; ld.global.u8 %r16,[%q18]; setp.ne.u32 %p6,%r15,%r16; @%p6 bra obj_next; add.u32 %r14,%r14,1; bra name_loop;
name_equal: add.u32 %r8,%r8,1; mov.u32 %r7,%r9; mov.u32 %r40,%r10;
obj_next: add.u64 %q13,%q13,1; bra obj_loop;
obj_done: setp.eq.u32 %p7,%r8,0; @%p7 bra missing; setp.ne.u32 %p8,%r8,1; @%p8 bra done;
 // Bootstrap and system policy are device policies after target resolution.  A non-bootstrap
 // principal must still resolve in the exact role generation before it can read a system relation.
 setp.ne.u32 %p9,%r1,1; @%p9 bra role_find; setp.ne.u32 %p10,%r0,10; @%p10 bra done; mov.u32 %r30,1; bra allow;
role_find: mov.u64 %q19,0; mov.u32 %r17,0; mov.u32 %r18,0; mov.u32 %r19,0;
role_loop: setp.ge.u64 %p11,%q19,%q2; @%p11 bra role_done; mul.lo.u64 %q20,%q19,12; add.u64 %q21,%q1,%q20; ld.global.u32 %r20,[%q21]; setp.ne.u32 %p12,%r20,%r0; @%p12 bra role_next; ld.global.u32 %r17,[%q21+4]; ld.global.u32 %r18,[%q21+8]; mov.u32 %r19,1;
role_next: add.u64 %q19,%q19,1; bra role_loop;
role_done: setp.eq.u32 %p13,%r19,0; @%p13 bra deny; setp.eq.u32 %p9,%r40,5; @%p9 bra allow;
 // Public schema existence itself is a candidate fact.  Do not convert an empty ACL list into
 // implicit USAGE when the schema was dropped in the pinned generation.
 mov.u64 %q33,0; mov.u32 %r41,0;
schema_object_loop: setp.ge.u64 %p14,%q33,%q4; @%p14 bra schema_object_done; mul.lo.u64 %q34,%q33,16; add.u64 %q35,%q3,%q34; ld.global.u32 %r42,[%q35+4]; ld.global.u32 %r43,[%q35+8]; ld.global.u32 %r44,[%q35+12]; setp.ne.u32 %p15,%r42,4; @%p15 bra schema_object_next; setp.ne.u32 %p16,%r44,6; @%p16 bra schema_object_next; add.u32 %r45,%r43,%r44; cvt.u64.u32 %q38,%r45; setp.gt.u64 %p17,%q38,%q10; @%p17 bra done; cvt.u64.u32 %q39,%r43; add.u64 %q39,%q9,%q39; ld.global.u8 %r45,[%q39]; ld.global.u8 %r46,[%q39+1]; ld.global.u8 %r47,[%q39+2]; setp.ne.u32 %p15,%r45,112; setp.ne.u32 %p16,%r46,117; or.pred %p15,%p15,%p16; setp.ne.u32 %p16,%r47,98; or.pred %p15,%p15,%p16; ld.global.u8 %r45,[%q39+3]; ld.global.u8 %r46,[%q39+4]; ld.global.u8 %r47,[%q39+5]; setp.ne.u32 %p16,%r45,108; or.pred %p15,%p15,%p16; setp.ne.u32 %p16,%r46,105; or.pred %p15,%p15,%p16; setp.ne.u32 %p16,%r47,99; or.pred %p15,%p15,%p16; @%p15 bra schema_object_next; mov.u32 %r41,1;
schema_object_next: add.u64 %q33,%q33,1; bra schema_object_loop;
schema_object_done: setp.eq.u32 %p14,%r41,0; @%p14 bra schema_deny;
 // Schema requests need CREATE; every user object request first needs public USAGE.
 setp.eq.u32 %p14,%r2,4; @%p14 bra schema_priv; mov.u32 %r21,1; bra schema_scan;
schema_priv: mov.u32 %r21,2;
schema_scan: mov.u64 %q22,0; mov.u32 %r22,0;
sgrant_loop: setp.ge.u64 %p15,%q22,%q8; @%p15 bra sgrant_done; mul.lo.u64 %q23,%q22,16; add.u64 %q24,%q7,%q23; ld.global.u32 %r23,[%q24+8]; setp.ne.u32 %p16,%r23,%r21; @%p16 bra sgrant_next; ld.global.u32 %r39,[%q24+12]; setp.eq.u32 %p0,%r39,1; @%p0 bra sgrant_allow; ld.global.u32 %r24,[%q24]; ld.global.u32 %r25,[%q24+4]; // direct grantee comparison
 add.u32 %r26,%r24,%r25; cvt.u64.u32 %q25,%r26; setp.gt.u64 %p17,%q25,%q10; @%p17 bra done; setp.ne.u32 %p0,%r25,%r18; @%p0 bra sgrant_next; mov.u32 %r27,0;
sname_loop: setp.ge.u32 %p2,%r27,%r18; @%p2 bra sgrant_allow; cvt.u64.u32 %q36,%r24; add.u64 %q26,%q9,%q36; cvt.u64.u32 %q37,%r27; add.u64 %q26,%q26,%q37; ld.global.u8 %r28,[%q26]; cvt.u64.u32 %q36,%r17; add.u64 %q27,%q9,%q36; add.u64 %q27,%q27,%q37; ld.global.u8 %r29,[%q27]; setp.ne.u32 %p3,%r28,%r29; @%p3 bra sgrant_next; add.u32 %r27,%r27,1; bra sname_loop;
sgrant_allow: mov.u32 %r22,1; bra sgrant_done;
sgrant_next: add.u64 %q22,%q22,1; bra sgrant_loop;
sgrant_done: setp.eq.u32 %p4,%r22,0; @%p4 bra schema_empty; bra schema_ok;
// PostgreSQL's empty public-schema ACL supplies implicit USAGE only.  CREATE always requires a
// matching device schema-grant row, including after the final CREATE grant is revoked.
schema_empty: setp.ne.u64 %p5,%q8,0; @%p5 bra schema_deny; setp.eq.u32 %p5,%r21,1; @%p5 bra schema_ok; bra schema_deny;
schema_ok: setp.eq.u32 %p6,%r2,4; @%p6 bra allow;
 // Object grant is a device join over target object OID/kind and principal/public grantee rows.
 mov.u64 %q28,0;
grant_loop: setp.ge.u64 %p7,%q28,%q6; @%p7 bra deny; mul.lo.u64 %q29,%q28,24; add.u64 %q30,%q5,%q29; ld.global.u32 %r31,[%q30]; ld.global.u32 %r32,[%q30+4]; ld.global.u32 %r33,[%q30+16]; setp.ne.u32 %p8,%r31,%r7; @%p8 bra grant_next; setp.ne.u32 %p9,%r32,%r2; @%p9 bra grant_next; setp.ne.u32 %p10,%r33,%r3; @%p10 bra grant_next; ld.global.u32 %r39,[%q30+20]; setp.eq.u32 %p11,%r39,1; @%p11 bra allow; ld.global.u32 %r34,[%q30+8]; ld.global.u32 %r35,[%q30+12]; setp.ne.u32 %p12,%r35,%r18; @%p12 bra grant_next;
grant_name: mov.u32 %r36,0;
gname_loop: setp.ge.u32 %p13,%r36,%r18; @%p13 bra allow; cvt.u64.u32 %q36,%r34; add.u64 %q31,%q9,%q36; cvt.u64.u32 %q37,%r36; add.u64 %q31,%q31,%q37; ld.global.u8 %r37,[%q31]; cvt.u64.u32 %q36,%r17; add.u64 %q32,%q9,%q36; add.u64 %q32,%q32,%q37; ld.global.u8 %r38,[%q32]; setp.ne.u32 %p14,%r37,%r38; @%p14 bra grant_next; add.u32 %r36,%r36,1; bra gname_loop;
grant_next: add.u64 %q28,%q28,1; bra grant_loop;
allow: st.global.u32 [%q11],1; bra done;
schema_deny: st.global.u32 [%q11],2; bra done;
deny: st.global.u32 [%q11],5; bra done;
missing: st.global.u32 [%q11],3;
done: ret; }
"#;

// Role: oid, name offset, name len.  Output: verdict (found/missing/invalid), stable OID.
// This is deliberately separate from the ACL operator because SET ROLE needs only name-to-OID
// resolution and must not inherit an authorization target as a surrogate predicate.
const ROLE_NAME_PTX: &[u8] = br#"
.version 6.0
.target sm_60
.address_size 64
.visible .entry gpu_db_catalog_role_name_verdict(
 .param .u64 reqp, .param .u64 rolesp, .param .u64 rolesn,
 .param .u64 bytesp, .param .u64 bytesn, .param .u64 outp) {
 .reg .pred %p<8>; .reg .u32 %r<24>; .reg .u64 %q<24>;
 ld.param.u64 %q0,[reqp]; ld.param.u64 %q1,[rolesp]; ld.param.u64 %q2,[rolesn];
 ld.param.u64 %q3,[bytesp]; ld.param.u64 %q4,[bytesn]; ld.param.u64 %q5,[outp];
 st.global.u32 [%q5],0; st.global.u32 [%q5+4],0;
 ld.global.u32 %r0,[%q0]; ld.global.u32 %r1,[%q0+4]; add.u32 %r2,%r0,%r1;
 cvt.u64.u32 %q6,%r2; setp.gt.u64 %p0,%q6,%q4; @%p0 bra done;
 mov.u64 %q7,0; mov.u32 %r3,0; mov.u32 %r4,0;
role_loop: setp.ge.u64 %p1,%q7,%q2; @%p1 bra role_done;
 mul.lo.u64 %q8,%q7,12; add.u64 %q9,%q1,%q8; ld.global.u32 %r5,[%q9]; ld.global.u32 %r6,[%q9+4]; ld.global.u32 %r7,[%q9+8];
 add.u32 %r8,%r6,%r7; cvt.u64.u32 %q10,%r8; setp.gt.u64 %p2,%q10,%q4; @%p2 bra done;
 setp.ne.u32 %p3,%r7,%r1; @%p3 bra role_next; mov.u32 %r9,0;
name_loop: setp.ge.u32 %p4,%r9,%r1; @%p4 bra name_equal;
 cvt.u64.u32 %q11,%r6; add.u64 %q12,%q3,%q11; cvt.u64.u32 %q13,%r9; add.u64 %q12,%q12,%q13; ld.global.u8 %r10,[%q12];
 cvt.u64.u32 %q14,%r0; add.u64 %q15,%q3,%q14; add.u64 %q15,%q15,%q13; ld.global.u8 %r11,[%q15];
 setp.ne.u32 %p5,%r10,%r11; @%p5 bra role_next; add.u32 %r9,%r9,1; bra name_loop;
name_equal: add.u32 %r3,%r3,1; mov.u32 %r4,%r5;
role_next: add.u64 %q7,%q7,1; bra role_loop;
role_done: setp.eq.u32 %p6,%r3,0; @%p6 bra missing; setp.ne.u32 %p7,%r3,1; @%p7 bra done;
 st.global.u32 [%q5],1; st.global.u32 [%q5+4],%r4; bra done;
missing: st.global.u32 [%q5],2;
done: ret; }
"#;

// Value: ordinal, name offset, name len, type oid, typmod.  Candidate: oid, display offset,
// display len.  Row layout is three u32s followed by two 256-byte cells.
const FORMAT_TYPE_PTX: &[u8] = br#"
.version 6.0
.target sm_60
.address_size 64
.visible .entry gpu_db_catalog_format_types(
 .param .u64 valp,.param .u64 valn,.param .u64 typep,.param .u64 typen,
 .param .u64 bytesp,.param .u64 bytesn,.param .u64 outp,.param .u64 verdictp) {
 .reg .pred %p<12>; .reg .u32 %r<38>; .reg .u64 %q<30>;
 ld.param.u64 %q0,[valp]; ld.param.u64 %q1,[valn]; ld.param.u64 %q2,[typep]; ld.param.u64 %q3,[typen]; ld.param.u64 %q4,[bytesp]; ld.param.u64 %q5,[bytesn]; ld.param.u64 %q6,[outp]; ld.param.u64 %q7,[verdictp]; st.global.u32 [%q7],0; mov.u64 %q8,0;
value_loop: setp.ge.u64 %p0,%q8,%q1; @%p0 bra done; mul.lo.u64 %q9,%q8,20; add.u64 %q10,%q0,%q9; ld.global.u32 %r0,[%q10]; ld.global.u32 %r1,[%q10+4]; ld.global.u32 %r2,[%q10+8]; ld.global.s32 %r3,[%q10+12]; ld.global.s32 %r4,[%q10+16]; add.u32 %r5,%r1,%r2; cvt.u64.u32 %q11,%r5; setp.gt.u64 %p1,%q11,%q5; @%p1 bra invalid; mov.u64 %q12,0; mov.u32 %r6,0; mov.u32 %r7,0; mov.u32 %r8,0;
type_loop: setp.ge.u64 %p2,%q12,%q3; @%p2 bra type_done; mul.lo.u64 %q13,%q12,12; add.u64 %q14,%q2,%q13; ld.global.s32 %r9,[%q14]; setp.ne.s32 %p3,%r9,%r3; @%p3 bra type_next; ld.global.u32 %r6,[%q14+4]; ld.global.u32 %r7,[%q14+8]; add.u32 %r8,%r8,1;
type_next: add.u64 %q12,%q12,1; bra type_loop;
type_done: setp.eq.u32 %p4,%r8,0; @%p4 bra unknown; setp.ne.u32 %p5,%r8,1; @%p5 bra invalid; add.u32 %r10,%r6,%r7; cvt.u64.u32 %q15,%r10; setp.gt.u64 %p6,%q15,%q5; @%p6 bra invalid;
 // Numeric is OID 1700.  Its typmod is validated and formatted in the terminal device row.
 setp.ne.s32 %p7,%r3,1700; @%p7 bra copy_base; setp.lt.s32 %p8,%r4,0; @%p8 bra copy_base; add.s32 %r11,%r4,-4; setp.lt.s32 %p9,%r11,0; @%p9 bra bad_typmod; shr.u32 %r12,%r11,16; and.b32 %r13,%r11,65535; setp.eq.u32 %p10,%r12,0; @%p10 bra bad_typmod; setp.gt.u32 %p11,%r12,38; @%p11 bra bad_typmod; setp.gt.u32 %p0,%r13,%r12; @%p0 bra bad_typmod; bra copy_numeric;
copy_base: setp.gt.u32 %p0,%r2,256; @%p0 bra invalid; setp.gt.u32 %p1,%r7,256; @%p1 bra invalid; mul.lo.u64 %q16,%q8,524; add.u64 %q17,%q6,%q16; st.global.u32 [%q17],%r0; st.global.u32 [%q17+4],%r2; st.global.u32 [%q17+8],%r7; mov.u32 %r14,0;
copy_name: setp.ge.u32 %p1,%r14,%r2; @%p1 bra copy_type_start; cvt.u64.u32 %q22,%r1; add.u64 %q18,%q4,%q22; cvt.u64.u32 %q23,%r14; add.u64 %q18,%q18,%q23; ld.global.u8 %r15,[%q18]; add.u64 %q19,%q17,12; add.u64 %q19,%q19,%q23; st.global.u8 [%q19],%r15; add.u32 %r14,%r14,1; bra copy_name;
copy_type_start: mov.u32 %r14,0;
copy_type: setp.ge.u32 %p2,%r14,%r7; @%p2 bra next_value; cvt.u64.u32 %q22,%r6; add.u64 %q20,%q4,%q22; cvt.u64.u32 %q23,%r14; add.u64 %q20,%q20,%q23; ld.global.u8 %r15,[%q20]; add.u64 %q21,%q17,268; add.u64 %q21,%q21,%q23; st.global.u8 [%q21],%r15; add.u32 %r14,%r14,1; bra copy_type;
copy_numeric: setp.gt.u32 %p3,%r2,256; @%p3 bra invalid; add.u32 %r14,%r7,5; setp.ge.u32 %p4,%r12,10; @%p4 add.u32 %r14,%r14,1; setp.ge.u32 %p5,%r13,10; @%p5 add.u32 %r14,%r14,1; setp.gt.u32 %p6,%r14,256; @%p6 bra invalid; mul.lo.u64 %q16,%q8,524; add.u64 %q17,%q6,%q16; st.global.u32 [%q17],%r0; st.global.u32 [%q17+4],%r2; st.global.u32 [%q17+8],%r14; mov.u32 %r15,0;
numeric_name: setp.ge.u32 %p5,%r15,%r2; @%p5 bra numeric_base; cvt.u64.u32 %q22,%r1; add.u64 %q18,%q4,%q22; cvt.u64.u32 %q23,%r15; add.u64 %q18,%q18,%q23; ld.global.u8 %r16,[%q18]; add.u64 %q19,%q17,12; add.u64 %q19,%q19,%q23; st.global.u8 [%q19],%r16; add.u32 %r15,%r15,1; bra numeric_name;
numeric_base: mov.u32 %r15,0;
numeric_copybase: setp.ge.u32 %p6,%r15,%r7; @%p6 bra numeric_suffix; cvt.u64.u32 %q22,%r6; add.u64 %q20,%q4,%q22; cvt.u64.u32 %q23,%r15; add.u64 %q20,%q20,%q23; ld.global.u8 %r16,[%q20]; add.u64 %q21,%q17,268; add.u64 %q21,%q21,%q23; st.global.u8 [%q21],%r16; add.u32 %r15,%r15,1; bra numeric_copybase;
numeric_suffix: add.u64 %q21,%q17,268; cvt.u64.u32 %q22,%r7; add.u64 %q21,%q21,%q22; mov.u32 %r16,40; st.global.u8 [%q21],%r16; add.u64 %q21,%q21,1; div.u32 %r17,%r12,10; rem.u32 %r18,%r12,10; setp.eq.u32 %p7,%r17,0; @%p7 bra prec_one; add.u32 %r17,%r17,48; st.global.u8 [%q21],%r17; add.u64 %q21,%q21,1;
prec_one: add.u32 %r18,%r18,48; st.global.u8 [%q21],%r18; add.u64 %q21,%q21,1; mov.u32 %r16,44; st.global.u8 [%q21],%r16; add.u64 %q21,%q21,1; div.u32 %r17,%r13,10; rem.u32 %r18,%r13,10; setp.eq.u32 %p8,%r17,0; @%p8 bra scale_one; add.u32 %r17,%r17,48; st.global.u8 [%q21],%r17; add.u64 %q21,%q21,1;
scale_one: add.u32 %r18,%r18,48; st.global.u8 [%q21],%r18; add.u64 %q21,%q21,1; mov.u32 %r16,41; st.global.u8 [%q21],%r16; bra next_value;
next_value: add.u64 %q8,%q8,1; bra value_loop;
unknown: st.global.u32 [%q7],1; bra done;
bad_typmod: st.global.u32 [%q7],2; bra done;
invalid: st.global.u32 [%q7],3;
done: ret; }
"#;

// Function: oid, namespace oid, owner oid, result type oid, language oid, name offset/name len,
// prokind, provolatile, proparallel, prosecdef (11 x u32).  Text candidates: oid, offset, len.
// Description: class oid, object oid, object sub-id, offset, len.  Grant: function oid,
// privilege, grantee offset/len, public identity.  Output row: ordinal, function oid, then 13
// (length + 256-byte cell) records.  This intentionally runs as one bounded catalog control
// operator: it is a data-parallel database engine's control-plane leaf, not a CPU fallback.
const FUNCTION_LIST_PTX: &[u8] = br#"
.version 6.0
.target sm_60
.address_size 64
.visible .entry gpu_db_catalog_function_list(
 .param .u64 reqp, .param .u64 funcp, .param .u64 funcn,
 .param .u64 nsp, .param .u64 nsn, .param .u64 typep, .param .u64 typen,
 .param .u64 ownerp, .param .u64 ownern, .param .u64 langp, .param .u64 langn,
 .param .u64 descp, .param .u64 descn, .param .u64 grantp, .param .u64 grantn,
 .param .u64 bytesp, .param .u64 bytesn, .param .u64 outp, .param .u64 headerp) {
 .reg .pred %p<32>; .reg .u32 %r<128>; .reg .u64 %q<80>;
 ld.param.u64 %q0,[reqp]; ld.param.u64 %q1,[funcp]; ld.param.u64 %q2,[funcn];
 ld.param.u64 %q3,[nsp]; ld.param.u64 %q4,[nsn]; ld.param.u64 %q5,[typep]; ld.param.u64 %q6,[typen];
 ld.param.u64 %q7,[ownerp]; ld.param.u64 %q8,[ownern]; ld.param.u64 %q9,[langp]; ld.param.u64 %q10,[langn];
 ld.param.u64 %q11,[descp]; ld.param.u64 %q12,[descn]; ld.param.u64 %q13,[grantp]; ld.param.u64 %q14,[grantn];
 ld.param.u64 %q15,[bytesp]; ld.param.u64 %q16,[bytesn]; ld.param.u64 %q17,[outp]; ld.param.u64 %q18,[headerp];
 st.global.u32 [%q18],1; st.global.u32 [%q18+4],0;
 ld.global.u32 %r0,[%q0]; ld.global.u32 %r1,[%q0+4]; ld.global.u32 %r2,[%q0+8];
 add.u32 %r3,%r0,%r1; cvt.u64.u32 %q19,%r3; setp.gt.u64 %p0,%q19,%q16; @%p0 bra invalid;
 // Select the next not-yet-emitted matching raw function by name on-device.  The catalog model
 // currently has one public namespace, but namespace/type/owner/language remain independent
 // typed candidate relations resolved below.
 mov.u32 %r4,0;
select_next: mov.u64 %q20,0; mov.u32 %r5,0; mov.u32 %r6,0; mov.u32 %r7,0; mov.u32 %r8,0; mov.u32 %r9,0;
func_scan: setp.ge.u64 %p1,%q20,%q2; @%p1 bra func_scan_done;
 mul.lo.u64 %q21,%q20,44; add.u64 %q22,%q1,%q21; ld.global.u32 %r10,[%q22]; ld.global.u32 %r11,[%q22+20]; ld.global.u32 %r12,[%q22+24];
 add.u32 %r13,%r11,%r12; cvt.u64.u32 %q23,%r13; setp.gt.u64 %p2,%q23,%q16; @%p2 bra invalid;
 // Optional exact psql identifier filter; text matching belongs to the terminal device program.
 setp.eq.u32 %p3,%r1,0; @%p3 bra filter_ok; setp.ne.u32 %p4,%r12,%r1; @%p4 bra func_next;
 mov.u32 %r14,0;
filter_loop: setp.ge.u32 %p5,%r14,%r1; @%p5 bra filter_ok;
 cvt.u64.u32 %q24,%r11; add.u64 %q25,%q15,%q24; cvt.u64.u32 %q26,%r14; add.u64 %q25,%q25,%q26; ld.global.u8 %r15,[%q25];
 cvt.u64.u32 %q27,%r0; add.u64 %q28,%q15,%q27; add.u64 %q28,%q28,%q26; ld.global.u8 %r16,[%q28]; setp.ne.u32 %p6,%r15,%r16; @%p6 bra func_next; add.u32 %r14,%r14,1; bra filter_loop;
filter_ok:
 // Do not choose an OID already emitted by this terminal sort.
 mov.u32 %r17,0;
used_loop: setp.ge.u32 %p7,%r17,%r4; @%p7 bra unused; mul.lo.u32 %r18,%r17,3388; cvt.u64.u32 %q29,%r18; add.u64 %q30,%q17,%q29; ld.global.u32 %r19,[%q30+4]; setp.eq.u32 %p8,%r19,%r10; @%p8 bra func_next; add.u32 %r17,%r17,1; bra used_loop;
unused:
 setp.eq.u32 %p9,%r5,0; @%p9 bra choose_current;
 // Compare raw names lexicographically.  A tie is malformed for this modeled pg_proc slice.
 min.u32 %r20,%r12,%r8; mov.u32 %r21,0;
name_compare: setp.ge.u32 %p10,%r21,%r20; @%p10 bra name_prefix;
 cvt.u64.u32 %q31,%r11; add.u64 %q32,%q15,%q31; cvt.u64.u32 %q33,%r21; add.u64 %q32,%q32,%q33; ld.global.u8 %r22,[%q32];
 cvt.u64.u32 %q34,%r7; add.u64 %q35,%q15,%q34; add.u64 %q35,%q35,%q33; ld.global.u8 %r23,[%q35]; setp.lt.u32 %p11,%r22,%r23; @%p11 bra choose_current; setp.gt.u32 %p12,%r22,%r23; @%p12 bra func_next; add.u32 %r21,%r21,1; bra name_compare;
name_prefix: setp.lt.u32 %p13,%r12,%r8; @%p13 bra choose_current; bra func_next;
choose_current: mov.u32 %r5,1; mov.u32 %r6,%r10; mov.u32 %r7,%r11; mov.u32 %r8,%r12; mov.u64 %q36,%q22;
func_next: add.u64 %q20,%q20,1; bra func_scan;
func_scan_done: setp.eq.u32 %p14,%r5,0; @%p14 bra complete;
 // Output row base and stable ordinal/OID.
 mul.lo.u32 %r24,%r4,3388; cvt.u64.u32 %q37,%r24; add.u64 %q38,%q17,%q37; st.global.u32 [%q38],%r4; st.global.u32 [%q38+4],%r6;
 // Namespace OID join, exactly one complete candidate.
 ld.global.u32 %r25,[%q36+4]; mov.u64 %q39,0; mov.u32 %r26,0; mov.u32 %r27,0; mov.u32 %r28,0;
nsp_scan: setp.ge.u64 %p15,%q39,%q4; @%p15 bra nsp_done; mul.lo.u64 %q40,%q39,12; add.u64 %q41,%q3,%q40; ld.global.u32 %r29,[%q41]; setp.ne.u32 %p16,%r29,%r25; @%p16 bra nsp_next; ld.global.u32 %r27,[%q41+4]; ld.global.u32 %r28,[%q41+8]; add.u32 %r26,%r26,1;
nsp_next: add.u64 %q39,%q39,1; bra nsp_scan;
nsp_done: setp.ne.u32 %p17,%r26,1; @%p17 bra invalid; add.u32 %r30,%r27,%r28; cvt.u64.u32 %q42,%r30; setp.gt.u64 %p18,%q42,%q16; @%p18 bra invalid;
 // cell 0 namespace
 add.u64 %q43,%q38,8; st.global.u32 [%q43],%r28; mov.u32 %r31,0;
copy_nsp: setp.ge.u32 %p19,%r31,%r28; @%p19 bra copy_name_start; cvt.u64.u32 %q44,%r27; add.u64 %q45,%q15,%q44; cvt.u64.u32 %q46,%r31; add.u64 %q45,%q45,%q46; ld.global.u8 %r32,[%q45]; add.u64 %q47,%q43,4; add.u64 %q47,%q47,%q46; st.global.u8 [%q47],%r32; add.u32 %r31,%r31,1; bra copy_nsp;
copy_name_start:
 // cell 1 raw proname
 add.u64 %q43,%q38,268; st.global.u32 [%q43],%r8; mov.u32 %r31,0;
copy_name: setp.ge.u32 %p19,%r31,%r8; @%p19 bra type_join_start; cvt.u64.u32 %q44,%r7; add.u64 %q45,%q15,%q44; cvt.u64.u32 %q46,%r31; add.u64 %q45,%q45,%q46; ld.global.u8 %r32,[%q45]; add.u64 %q47,%q43,4; add.u64 %q47,%q47,%q46; st.global.u8 [%q47],%r32; add.u32 %r31,%r31,1; bra copy_name;
type_join_start:
 // Result type OID join.
 ld.global.u32 %r25,[%q36+12]; mov.u64 %q39,0; mov.u32 %r26,0; mov.u32 %r27,0; mov.u32 %r28,0;
type_scan: setp.ge.u64 %p15,%q39,%q6; @%p15 bra type_done; mul.lo.u64 %q40,%q39,12; add.u64 %q41,%q5,%q40; ld.global.u32 %r29,[%q41]; setp.ne.u32 %p16,%r29,%r25; @%p16 bra type_next; ld.global.u32 %r27,[%q41+4]; ld.global.u32 %r28,[%q41+8]; add.u32 %r26,%r26,1;
type_next: add.u64 %q39,%q39,1; bra type_scan;
type_done: setp.ne.u32 %p17,%r26,1; @%p17 bra invalid; add.u32 %r30,%r27,%r28; cvt.u64.u32 %q42,%r30; setp.gt.u64 %p18,%q42,%q16; @%p18 bra invalid;
 // cell 2 result display
 add.u64 %q43,%q38,528; st.global.u32 [%q43],%r28; mov.u32 %r31,0;
copy_type: setp.ge.u32 %p19,%r31,%r28; @%p19 bra fixed_fields; cvt.u64.u32 %q44,%r27; add.u64 %q45,%q15,%q44; cvt.u64.u32 %q46,%r31; add.u64 %q45,%q45,%q46; ld.global.u8 %r32,[%q45]; add.u64 %q47,%q43,4; add.u64 %q47,%q47,%q46; st.global.u8 [%q47],%r32; add.u32 %r31,%r31,1; bra copy_type;
fixed_fields:
 // Empty arguments (cell 3) and the device interpretation of raw prokind f -> func (cell 4).
 add.u64 %q43,%q38,788; st.global.u32 [%q43],0; ld.global.u32 %r33,[%q36+28]; setp.ne.u32 %p20,%r33,102; @%p20 bra invalid; add.u64 %q43,%q38,1048; st.global.u32 [%q43],4; add.u64 %q44,%q43,4; mov.u32 %r34,102; st.global.u8 [%q44],%r34; mov.u32 %r34,117; st.global.u8 [%q44+1],%r34; mov.u32 %r34,110; st.global.u8 [%q44+2],%r34; mov.u32 %r34,99; st.global.u8 [%q44+3],%r34;
 setp.eq.u32 %p21,%r2,0; @%p21 bra row_done;
 // Volatility and parallel are mapped from raw pg_proc code points by the terminal program.
 ld.global.u32 %r33,[%q36+32]; setp.ne.u32 %p20,%r33,118; @%p20 bra invalid; add.u64 %q43,%q38,1308; st.global.u32 [%q43],8; add.u64 %q44,%q43,4; mov.u32 %r34,118; st.global.u8 [%q44],%r34; mov.u32 %r34,111; st.global.u8 [%q44+1],%r34; mov.u32 %r34,108; st.global.u8 [%q44+2],%r34; mov.u32 %r34,97; st.global.u8 [%q44+3],%r34; mov.u32 %r34,116; st.global.u8 [%q44+4],%r34; mov.u32 %r34,105; st.global.u8 [%q44+5],%r34; mov.u32 %r34,108; st.global.u8 [%q44+6],%r34; mov.u32 %r34,101; st.global.u8 [%q44+7],%r34;
 ld.global.u32 %r33,[%q36+36]; setp.ne.u32 %p20,%r33,117; @%p20 bra invalid; add.u64 %q43,%q38,1568; st.global.u32 [%q43],6; add.u64 %q44,%q43,4; mov.u32 %r34,117; st.global.u8 [%q44],%r34; mov.u32 %r34,110; st.global.u8 [%q44+1],%r34; mov.u32 %r34,115; st.global.u8 [%q44+2],%r34; mov.u32 %r34,97; st.global.u8 [%q44+3],%r34; mov.u32 %r34,102; st.global.u8 [%q44+4],%r34; mov.u32 %r34,101; st.global.u8 [%q44+5],%r34;
 // Owner OID join (cell 7).
 ld.global.u32 %r25,[%q36+8]; mov.u64 %q39,0; mov.u32 %r26,0; mov.u32 %r27,0; mov.u32 %r28,0;
owner_scan: setp.ge.u64 %p15,%q39,%q8; @%p15 bra owner_done; mul.lo.u64 %q40,%q39,12; add.u64 %q41,%q7,%q40; ld.global.u32 %r29,[%q41]; setp.ne.u32 %p16,%r29,%r25; @%p16 bra owner_next; ld.global.u32 %r27,[%q41+4]; ld.global.u32 %r28,[%q41+8]; add.u32 %r26,%r26,1;
owner_next: add.u64 %q39,%q39,1; bra owner_scan;
owner_done: setp.ne.u32 %p17,%r26,1; @%p17 bra invalid; add.u32 %r30,%r27,%r28; cvt.u64.u32 %q42,%r30; setp.gt.u64 %p18,%q42,%q16; @%p18 bra invalid; add.u64 %q43,%q38,1828; st.global.u32 [%q43],%r28; mov.u32 %r31,0;
copy_owner: setp.ge.u32 %p19,%r31,%r28; @%p19 bra security_field; cvt.u64.u32 %q44,%r27; add.u64 %q45,%q15,%q44; cvt.u64.u32 %q46,%r31; add.u64 %q45,%q45,%q46; ld.global.u8 %r32,[%q45]; add.u64 %q47,%q43,4; add.u64 %q47,%q47,%q46; st.global.u8 [%q47],%r32; add.u32 %r31,%r31,1; bra copy_owner;
security_field:
 // Raw prosecdef controls the device-formatted Security cell.
 ld.global.u32 %r33,[%q36+40]; add.u64 %q43,%q38,2088; setp.eq.u32 %p20,%r33,0; @%p20 bra invoker; setp.eq.u32 %p20,%r33,1; @%p20 bra definer; bra invalid;
invoker: st.global.u32 [%q43],7; add.u64 %q44,%q43,4; mov.u32 %r34,105; st.global.u8 [%q44],%r34; mov.u32 %r34,110; st.global.u8 [%q44+1],%r34; mov.u32 %r34,118; st.global.u8 [%q44+2],%r34; mov.u32 %r34,111; st.global.u8 [%q44+3],%r34; mov.u32 %r34,107; st.global.u8 [%q44+4],%r34; mov.u32 %r34,101; st.global.u8 [%q44+5],%r34; mov.u32 %r34,114; st.global.u8 [%q44+6],%r34; bra acl_field;
definer: st.global.u32 [%q43],7; add.u64 %q44,%q43,4; mov.u32 %r34,100; st.global.u8 [%q44],%r34; mov.u32 %r34,101; st.global.u8 [%q44+1],%r34; mov.u32 %r34,102; st.global.u8 [%q44+2],%r34; mov.u32 %r34,105; st.global.u8 [%q44+3],%r34; mov.u32 %r34,110; st.global.u8 [%q44+4],%r34; mov.u32 %r34,101; st.global.u8 [%q44+5],%r34; mov.u32 %r34,114; st.global.u8 [%q44+6],%r34;
acl_field:
 // ACL display is sorted by effective grantee (PUBLIC is the empty first key) inside this
 // terminal program.  A duplicate matching raw grant is malformed and fails closed rather than
 // becoming a duplicate visible ACL entry or inheriting host BTreeMap iteration order.
 add.u64 %q43,%q38,2348; st.global.u32 [%q43],0; mov.u32 %r31,0; mov.u32 %r35,0;
 // r35/r36/r37/r38 = have-last, public, offset, len.  r39/r40/r41/r42/q48 = best candidate.
 mov.u32 %r36,0; mov.u32 %r37,0; mov.u32 %r38,0;
grant_select: mov.u64 %q39,0; mov.u32 %r39,0;
grant_scan: setp.ge.u64 %p15,%q39,%q14; @%p15 bra grant_scan_done; mul.lo.u64 %q40,%q39,20; add.u64 %q41,%q13,%q40; ld.global.u32 %r29,[%q41]; setp.ne.u32 %p16,%r29,%r6; @%p16 bra grant_next; ld.global.u32 %r29,[%q41+4]; setp.ne.u32 %p16,%r29,1; @%p16 bra grant_next; ld.global.u32 %r44,[%q41+8]; ld.global.u32 %r45,[%q41+12]; add.u32 %r30,%r44,%r45; cvt.u64.u32 %q42,%r30; setp.gt.u64 %p18,%q42,%q16; @%p18 bra invalid; ld.global.u32 %r43,[%q41+16];
 // A candidate must be strictly after the last emitted effective grantee key.
 setp.eq.u32 %p20,%r35,0; @%p20 bra grant_after_last;
 setp.eq.u32 %p21,%r36,1; @%p21 bra last_public;
 setp.eq.u32 %p22,%r43,1; @%p22 bra grant_next; min.u32 %r32,%r45,%r38; mov.u32 %r33,0;
last_compare: setp.ge.u32 %p23,%r33,%r32; @%p23 bra last_prefix; cvt.u64.u32 %q44,%r44; add.u64 %q45,%q15,%q44; cvt.u64.u32 %q46,%r33; add.u64 %q45,%q45,%q46; ld.global.u8 %r46,[%q45]; cvt.u64.u32 %q47,%r37; add.u64 %q44,%q15,%q47; add.u64 %q44,%q44,%q46; ld.global.u8 %r47,[%q44]; setp.gt.u32 %p24,%r46,%r47; @%p24 bra grant_after_last; setp.lt.u32 %p25,%r46,%r47; @%p25 bra grant_next; add.u32 %r33,%r33,1; bra last_compare;
last_prefix: setp.gt.u32 %p24,%r45,%r38; @%p24 bra grant_after_last; bra grant_next;
last_public: setp.eq.u32 %p22,%r43,1; @%p22 bra grant_next;
grant_after_last:
 // Retain the least eligible candidate.  Equal keys are duplicate raw grants and fail closed.
 setp.eq.u32 %p20,%r39,0; @%p20 bra grant_choose;
 setp.eq.u32 %p21,%r40,1; @%p21 bra best_public; setp.eq.u32 %p22,%r43,1; @%p22 bra grant_choose; min.u32 %r32,%r45,%r42; mov.u32 %r33,0;
best_compare: setp.ge.u32 %p23,%r33,%r32; @%p23 bra best_prefix; cvt.u64.u32 %q44,%r44; add.u64 %q45,%q15,%q44; cvt.u64.u32 %q46,%r33; add.u64 %q45,%q45,%q46; ld.global.u8 %r46,[%q45]; cvt.u64.u32 %q47,%r41; add.u64 %q44,%q15,%q47; add.u64 %q44,%q44,%q46; ld.global.u8 %r47,[%q44]; setp.lt.u32 %p24,%r46,%r47; @%p24 bra grant_choose; setp.gt.u32 %p25,%r46,%r47; @%p25 bra grant_next; add.u32 %r33,%r33,1; bra best_compare;
best_prefix: setp.lt.u32 %p24,%r45,%r42; @%p24 bra grant_choose; setp.eq.u32 %p25,%r45,%r42; @%p25 bra invalid; bra grant_next;
best_public: setp.eq.u32 %p22,%r43,1; @%p22 bra invalid; bra grant_next;
grant_choose: mov.u32 %r39,1; mov.u32 %r40,%r43; mov.u32 %r41,%r44; mov.u32 %r42,%r45; mov.u64 %q48,%q41;
grant_next: add.u64 %q39,%q39,1; bra grant_scan;
grant_scan_done: setp.eq.u32 %p20,%r39,0; @%p20 bra grant_done;
 // Append the selected raw candidate.  PUBLIC has the empty display key but retains its typed
 // raw identity in the ordering/dedup checks above.
 setp.eq.u32 %p20,%r35,0; @%p20 bra grant_emit_name; add.u32 %r31,%r31,1; setp.gt.u32 %p20,%r31,256; @%p20 bra invalid; add.u64 %q44,%q43,4; cvt.u64.u32 %q45,%r31; add.u64 %q44,%q44,%q45; add.u64 %q44,%q44,-1; mov.u32 %r34,10; st.global.u8 [%q44],%r34;
grant_emit_name: setp.eq.u32 %p20,%r40,1; @%p20 bra grant_suffix; mov.u32 %r32,0;
grant_name_copy: setp.ge.u32 %p19,%r32,%r42; @%p19 bra grant_suffix; add.u32 %r31,%r31,1; setp.gt.u32 %p20,%r31,256; @%p20 bra invalid; cvt.u64.u32 %q44,%r41; add.u64 %q45,%q15,%q44; cvt.u64.u32 %q46,%r32; add.u64 %q45,%q45,%q46; ld.global.u8 %r34,[%q45]; add.u64 %q47,%q43,4; cvt.u64.u32 %q44,%r31; add.u64 %q47,%q47,%q44; add.u64 %q47,%q47,-1; st.global.u8 [%q47],%r34; add.u32 %r32,%r32,1; bra grant_name_copy;
grant_suffix: add.u32 %r31,%r31,11; setp.gt.u32 %p20,%r31,256; @%p20 bra invalid; add.u64 %q44,%q43,4; cvt.u64.u32 %q45,%r31; add.u64 %q44,%q44,%q45; add.u64 %q44,%q44,-11; mov.u32 %r34,61; st.global.u8 [%q44],%r34; mov.u32 %r34,88; st.global.u8 [%q44+1],%r34; mov.u32 %r34,47; st.global.u8 [%q44+2],%r34; mov.u32 %r34,112; st.global.u8 [%q44+3],%r34; mov.u32 %r34,111; st.global.u8 [%q44+4],%r34; mov.u32 %r34,115; st.global.u8 [%q44+5],%r34; mov.u32 %r34,116; st.global.u8 [%q44+6],%r34; mov.u32 %r34,103; st.global.u8 [%q44+7],%r34; mov.u32 %r34,114; st.global.u8 [%q44+8],%r34; mov.u32 %r34,101; st.global.u8 [%q44+9],%r34; mov.u32 %r34,115; st.global.u8 [%q44+10],%r34; mov.u32 %r35,1; mov.u32 %r36,%r40; mov.u32 %r37,%r41; mov.u32 %r38,%r42; bra grant_select;
grant_done: setp.ne.u32 %p20,%r35,0; @%p20 bra acl_done; mov.u32 %r34,4294967295; st.global.u32 [%q43],%r34;
acl_done: setp.ne.u32 %p20,%r35,0; @%p20 st.global.u32 [%q43],%r31;
 // Language OID join (cell 10).
 ld.global.u32 %r25,[%q36+16]; mov.u64 %q39,0; mov.u32 %r26,0; mov.u32 %r27,0; mov.u32 %r28,0;
lang_scan: setp.ge.u64 %p15,%q39,%q10; @%p15 bra lang_done; mul.lo.u64 %q40,%q39,12; add.u64 %q41,%q9,%q40; ld.global.u32 %r29,[%q41]; setp.ne.u32 %p16,%r29,%r25; @%p16 bra lang_next; ld.global.u32 %r27,[%q41+4]; ld.global.u32 %r28,[%q41+8]; add.u32 %r26,%r26,1;
lang_next: add.u64 %q39,%q39,1; bra lang_scan;
lang_done: setp.ne.u32 %p17,%r26,1; @%p17 bra invalid; add.u32 %r30,%r27,%r28; cvt.u64.u32 %q42,%r30; setp.gt.u64 %p18,%q42,%q16; @%p18 bra invalid; add.u64 %q43,%q38,2608; st.global.u32 [%q43],%r28; mov.u32 %r31,0;
copy_lang: setp.ge.u32 %p19,%r31,%r28; @%p19 bra internal_null; cvt.u64.u32 %q44,%r27; add.u64 %q45,%q15,%q44; cvt.u64.u32 %q46,%r31; add.u64 %q45,%q45,%q46; ld.global.u8 %r32,[%q45]; add.u64 %q47,%q43,4; add.u64 %q47,%q47,%q46; st.global.u8 [%q47],%r32; add.u32 %r31,%r31,1; bra copy_lang;
internal_null: add.u64 %q43,%q38,2868; mov.u32 %r34,4294967295; st.global.u32 [%q43],%r34;
 // Raw pg_description join on pg_proc class and objsubid 0 (cell 12); no matching row is SQL NULL.
 mov.u64 %q39,0; mov.u32 %r26,0; mov.u32 %r27,0; mov.u32 %r28,0;
desc_scan: setp.ge.u64 %p15,%q39,%q12; @%p15 bra desc_done; mul.lo.u64 %q40,%q39,20; add.u64 %q41,%q11,%q40; ld.global.s32 %r29,[%q41]; setp.ne.s32 %p16,%r29,1255; @%p16 bra desc_next; ld.global.u32 %r29,[%q41+4]; setp.ne.u32 %p16,%r29,%r6; @%p16 bra desc_next; ld.global.s32 %r29,[%q41+8]; setp.ne.s32 %p16,%r29,0; @%p16 bra desc_next; ld.global.u32 %r27,[%q41+12]; ld.global.u32 %r28,[%q41+16]; add.u32 %r26,%r26,1;
desc_next: add.u64 %q39,%q39,1; bra desc_scan;
desc_done: setp.gt.u32 %p20,%r26,1; @%p20 bra invalid; add.u64 %q43,%q38,3128; setp.eq.u32 %p20,%r26,0; @%p20 bra desc_null; add.u32 %r30,%r27,%r28; cvt.u64.u32 %q42,%r30; setp.gt.u64 %p18,%q42,%q16; @%p18 bra invalid; st.global.u32 [%q43],%r28; mov.u32 %r31,0;
copy_desc: setp.ge.u32 %p19,%r31,%r28; @%p19 bra row_done; cvt.u64.u32 %q44,%r27; add.u64 %q45,%q15,%q44; cvt.u64.u32 %q46,%r31; add.u64 %q45,%q45,%q46; ld.global.u8 %r32,[%q45]; add.u64 %q47,%q43,4; add.u64 %q47,%q47,%q46; st.global.u8 [%q47],%r32; add.u32 %r31,%r31,1; bra copy_desc;
desc_null: mov.u32 %r34,4294967295; st.global.u32 [%q43],%r34;
row_done: add.u32 %r4,%r4,1; bra select_next;
complete: st.global.u32 [%q18],0; st.global.u32 [%q18+4],%r4; ret;
invalid: st.global.u32 [%q18],1; st.global.u32 [%q18+4],0; ret; }
"#;
