use std::ffi::{CStr, c_char};

// Exported by the Swift package in `swift/`. Each returns a JSON response that the caller frees
// with `koharu_apple_free`, or null when the response could not be encoded.
unsafe extern "C" {
    fn koharu_apple_recognize_text(
        image: *const u8,
        length: isize,
        languages: *const c_char,
    ) -> *mut c_char;
    fn koharu_apple_translate(request: *const c_char) -> *mut c_char;
    fn koharu_apple_free(pointer: *mut c_char);
}

pub(crate) fn recognize_text(image: &[u8], languages: &CStr) -> Option<String> {
    // SAFETY: The image slice and the language string outlive the synchronous call.
    unsafe {
        take(koharu_apple_recognize_text(
            image.as_ptr(),
            image.len() as isize,
            languages.as_ptr(),
        ))
    }
}

pub(crate) fn translate(request: &CStr) -> Option<String> {
    // SAFETY: The request string outlives the synchronous call.
    unsafe { take(koharu_apple_translate(request.as_ptr())) }
}

/// Copies a Swift-owned response and releases it.
unsafe fn take(pointer: *mut c_char) -> Option<String> {
    if pointer.is_null() {
        return None;
    }
    // SAFETY: Non-null responses are NUL-terminated strings allocated by the Swift package.
    let response = unsafe { CStr::from_ptr(pointer) }
        .to_string_lossy()
        .into_owned();
    unsafe { koharu_apple_free(pointer) };
    Some(response)
}
