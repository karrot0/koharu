//! Compiled Core ML models (`.mlmodelc`) on iOS, where Koharu ships no Torch runtime.
//!
//! Core ML places each model on the Neural Engine, GPU, or CPU itself. Tensors cross the boundary
//! as dense float32 multi-arrays; outputs are read in place, so a caller can copy out only the
//! slices it keeps.

use std::{cell::RefCell, ffi::c_void, path::Path, ptr::NonNull};

use anyhow::{Context, Result, anyhow, bail, ensure};
use block2::StackBlock;
use objc2::{
    AllocAnyThread,
    rc::{Retained, autoreleasepool},
    runtime::{AnyObject, ProtocolObject},
};
use objc2_core_ml::{
    MLComputeUnits, MLDictionaryFeatureProvider, MLFeatureProvider, MLFeatureValue, MLModel,
    MLModelConfiguration, MLMultiArray, MLMultiArrayDataType,
};
use objc2_foundation::{NSArray, NSDictionary, NSNumber, NSString, NSURL};

/// A compiled Core ML model.
#[derive(Debug)]
pub(crate) struct CoreMlModel {
    model: Retained<MLModel>,
}

// SAFETY: Core ML accepts predictions from any thread, and Koharu serializes each model's
// predictions behind its stage lock.
unsafe impl Send for CoreMlModel {}
unsafe impl Sync for CoreMlModel {}

impl CoreMlModel {
    pub(crate) fn load(path: &Path) -> Result<Self> {
        let url = NSURL::from_directory_path(path)
            .with_context(|| format!("invalid Core ML model path {}", path.display()))?;
        autoreleasepool(|_| {
            let configuration = unsafe { MLModelConfiguration::new() };
            unsafe { configuration.setComputeUnits(MLComputeUnits::All) };
            let model = unsafe {
                MLModel::modelWithContentsOfURL_configuration_error(&url, &configuration)
            }
            .map_err(|error| anyhow!("failed to load {}: {error}", path.display()))?;
            Ok(Self { model })
        })
    }

    /// Runs the model on named float32 inputs and returns the named outputs in order.
    pub(crate) fn predict(
        &self,
        inputs: &[(&str, &Tensor)],
        outputs: &[&str],
    ) -> Result<Vec<Array>> {
        autoreleasepool(|_| {
            let mut names = Vec::with_capacity(inputs.len());
            let mut values = Vec::<Retained<AnyObject>>::with_capacity(inputs.len());
            for (name, tensor) in inputs {
                let array = multi_array(tensor)?;
                let value = unsafe { MLFeatureValue::featureValueWithMultiArray(&array) };
                names.push(NSString::from_str(name));
                values.push(Retained::into_super(Retained::into_super(value)));
            }
            let keys = names.iter().map(|name| &**name).collect::<Vec<_>>();
            let dictionary = NSDictionary::from_retained_objects(&keys, &values);
            let provider = unsafe {
                MLDictionaryFeatureProvider::initWithDictionary_error(
                    MLDictionaryFeatureProvider::alloc(),
                    &dictionary,
                )
            }
            .map_err(|error| anyhow!("invalid Core ML input: {error}"))?;
            let prediction = unsafe {
                self.model
                    .predictionFromFeatures_error(ProtocolObject::from_ref(&*provider))
            }
            .map_err(|error| anyhow!("Core ML prediction failed: {error}"))?;
            outputs
                .iter()
                .map(|name| {
                    let value =
                        unsafe { prediction.featureValueForName(&NSString::from_str(name)) }
                            .with_context(|| format!("Core ML output {name} is missing"))?;
                    let array = unsafe { value.multiArrayValue() }
                        .with_context(|| format!("Core ML output {name} is not a multi-array"))?;
                    Array::new(array)
                })
                .collect()
        })
    }
}

/// A dense, row-major float32 tensor sent to Core ML.
#[derive(Debug)]
pub(crate) struct Tensor {
    pub(crate) shape: Vec<usize>,
    pub(crate) data: Vec<f32>,
}

/// A Core ML output, read in place.
pub(crate) struct Array {
    array: Retained<MLMultiArray>,
    shape: Vec<usize>,
    strides: Vec<usize>,
    element: Element,
}

#[derive(Clone, Copy)]
enum Element {
    Float32,
    Float16,
}

impl Array {
    fn new(array: Retained<MLMultiArray>) -> Result<Self> {
        let element = match unsafe { array.dataType() } {
            MLMultiArrayDataType::Float32 => Element::Float32,
            MLMultiArrayDataType::Float16 => Element::Float16,
            other => bail!("unsupported Core ML output type {other:?}"),
        };
        Ok(Self {
            shape: numbers(&*unsafe { array.shape() })?,
            strides: numbers(&*unsafe { array.strides() })?,
            array,
            element,
        })
    }

    pub(crate) fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Copies the sub-tensor at the leading `index` (empty for the whole array) as row-major
    /// float32.
    pub(crate) fn read(&self, index: &[usize]) -> Result<Vec<f32>> {
        ensure!(
            index.len() <= self.shape.len()
                && index
                    .iter()
                    .zip(&self.shape)
                    .all(|(index, size)| index < size),
            "index {index:?} is outside Core ML output shape {:?}",
            self.shape
        );
        let base = index
            .iter()
            .zip(&self.strides)
            .map(|(index, stride)| index * stride)
            .sum::<usize>();
        let shape = &self.shape[index.len()..];
        let strides = &self.strides[index.len()..];
        let result = RefCell::new(None);
        let block = StackBlock::new(|bytes: NonNull<c_void>, size: isize| {
            // SAFETY: Core ML lends `size` readable bytes for the duration of the handler.
            let bytes =
                unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<u8>(), size as usize) };
            *result.borrow_mut() = Some(gather(bytes, self.element, base, shape, strides));
        });
        unsafe { self.array.getBytesWithHandler(&block) };
        result
            .into_inner()
            .context("Core ML did not lend its output buffer")?
    }
}

fn gather(
    bytes: &[u8],
    element: Element,
    base: usize,
    shape: &[usize],
    strides: &[usize],
) -> Result<Vec<f32>> {
    let width = match element {
        Element::Float32 => 4,
        Element::Float16 => 2,
    };
    let scalar = |offset: usize| -> Result<f32> {
        let start = offset * width;
        let raw = bytes
            .get(start..start + width)
            .context("Core ML output is shorter than its shape")?;
        Ok(match element {
            Element::Float32 => f32::from_ne_bytes(raw.try_into()?),
            Element::Float16 => half::f16::from_ne_bytes(raw.try_into()?).to_f32(),
        })
    };
    let count = shape.iter().product::<usize>();
    let mut values = Vec::with_capacity(count);
    let mut index = vec![0; shape.len()];
    for _ in 0..count {
        let offset = base
            + index
                .iter()
                .zip(strides)
                .map(|(index, stride)| index * stride)
                .sum::<usize>();
        values.push(scalar(offset)?);
        for axis in (0..shape.len()).rev() {
            index[axis] += 1;
            if index[axis] < shape[axis] {
                break;
            }
            index[axis] = 0;
        }
    }
    Ok(values)
}

fn multi_array(tensor: &Tensor) -> Result<Retained<MLMultiArray>> {
    ensure!(
        tensor.shape.iter().product::<usize>() == tensor.data.len(),
        "tensor data does not match its shape {:?}",
        tensor.shape
    );
    let shape = tensor
        .shape
        .iter()
        .map(|size| NSNumber::new_isize(*size as isize))
        .collect::<Vec<_>>();
    let array = unsafe {
        MLMultiArray::initWithShape_dataType_error(
            MLMultiArray::alloc(),
            &NSArray::from_retained_slice(&shape),
            MLMultiArrayDataType::Float32,
        )
    }
    .map_err(|error| anyhow!("failed to allocate a Core ML input: {error}"))?;
    // `initWithShape:dataType:error:` lays the scalars out contiguously, first dimension major.
    let block = StackBlock::new(
        |bytes: NonNull<c_void>, size: isize, _strides: NonNull<NSArray<NSNumber>>| {
            let count = (size as usize / size_of::<f32>()).min(tensor.data.len());
            // SAFETY: Core ML lends `size` writable bytes for the duration of the handler.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    tensor.data.as_ptr(),
                    bytes.as_ptr().cast::<f32>(),
                    count,
                );
            }
        },
    );
    unsafe { array.getMutableBytesWithHandler(&block) };
    Ok(array)
}

fn numbers(array: &NSArray<NSNumber>) -> Result<Vec<usize>> {
    array
        .iter()
        .map(|number| usize::try_from(number.as_isize()).context("negative Core ML dimension"))
        .collect()
}
