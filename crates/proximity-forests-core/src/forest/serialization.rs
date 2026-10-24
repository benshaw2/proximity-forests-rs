//! Stable, versioned model serialization framing and DTO schema.
//!
//! Runtime model structs are deliberately not serialized directly. Each format
//! version owns a separate DTO graph, allowing runtime refactors without
//! silently changing persisted bytes.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::str::FromStr;
use std::{error::Error, fmt};

/// Eight-byte file marker: `PFOREST\0`.
pub(crate) const MODEL_MAGIC: [u8; 8] = *b"PFOREST\0";
/// Current envelope version.
pub(crate) const MODEL_FORMAT_VERSION: u32 = 1;
const HEADER_LEN: usize = 8 + 4 + 1 + 8 + 4;

/// Persisted estimator family.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[repr(u8)]
pub(crate) enum ModelKindDto {
    Classification = 1,
    Regression = 2,
    Isolation = 3,
}

impl TryFrom<u8> for ModelKindDto {
    type Error = SerializationError;
    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::Classification),
            2 => Ok(Self::Regression),
            3 => Ok(Self::Isolation),
            other => Err(SerializationError::UnknownModelKind(other)),
        }
    }
}

/// Version-one task DTO. The task-specific body is a separately encoded DTO
/// graph; it is never a byte dump of a runtime Rust struct.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub(crate) enum ModelDtoV1 {
    Classification(ClassificationModelDtoV1),
    Regression(RegressionModelDtoV1),
    Isolation(IsolationModelDtoV1),
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub(crate) struct ClassificationModelDtoV1 {
    pub(crate) forest: ForestDtoV1,
    pub(crate) output_count: u32,
}
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub(crate) struct RegressionModelDtoV1 {
    pub(crate) forest: ForestDtoV1,
    pub(crate) output_count: u32,
}
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub(crate) struct IsolationModelDtoV1 {
    pub(crate) forest: ForestDtoV1,
    pub(crate) normalization_size: u64,
}

/// Stable top-level forest DTO. Additional graph DTOs are intentionally stored
/// in the versioned body rather than exposing runtime object layout.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub(crate) struct ForestDtoV1 {
    pub(crate) config: ForestConfigDtoV1,
    pub(crate) trees: Vec<TreeDtoV1>,
    pub(crate) retained_training_data: RetainedTrainingDataDtoV1,
    pub(crate) training_metadata: TrainingMetadataDtoV1,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub(crate) struct ForestConfigDtoV1 {
    pub(crate) tree_count: u64,
    pub(crate) candidate_count: u64,
    pub(crate) bootstrap: bool,
    pub(crate) maximum_depth: Option<u64>,
    pub(crate) minimum_split_size: u64,
    pub(crate) dimension_selection: DimensionSelectionDtoV1,
    pub(crate) seed: u64,
    pub(crate) early_abandoning: bool,
    pub(crate) branch_tie_policy: TiePolicyDtoV1,
    pub(crate) data_retention: DataRetentionDtoV1,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub(crate) enum TiePolicyDtoV1 {
    Random,
    First,
}
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub(crate) enum DataRetentionDtoV1 {
    Exemplars,
    Copy,
    Reference,
}
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub(crate) enum DimensionSelectionDtoV1 {
    All,
    FixedCount(u64),
    SquareRoot,
    Log2,
    Fraction(f64),
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub(crate) struct TreeDtoV1 {
    pub(crate) root: u64,
    pub(crate) nodes: Vec<TreeNodeDtoV1>,
    pub(crate) exemplars: ObservationStoreDtoV1,
}
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub(crate) enum TreeNodeDtoV1 {
    Split {
        distance: DistanceConfigDtoV1,
        selected_dimensions: Vec<u64>,
        branches: Vec<BranchDtoV1>,
    },
    Leaf {
        payload: LeafPayloadDtoV1,
        occurrence_count: u64,
        distinct_count: u64,
    },
}
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub(crate) struct BranchDtoV1 {
    pub exemplar_id: u32,
    pub child: u64,
}

/// Distance DTO uses a stable symbolic family plus named numeric/integer
/// parameters. Names are part of format v1 and permit future runtime type
/// changes without coupling the file to Rust enum discriminants.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub(crate) struct DistanceConfigDtoV1 {
    pub(crate) family: String,
    pub(crate) floats: Vec<(String, f64)>,
    pub(crate) integers: Vec<(String, u64)>,
    pub(crate) flags: Vec<(String, bool)>,
    pub(crate) strings: Vec<(String, String)>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub(crate) enum LeafPayloadDtoV1 {
    Empty,
    Classification { counts: Vec<Vec<(u32, u64)>> },
    Regression { means: Vec<f64> },
    Isolation { occurrence_count: u64 },
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub(crate) enum ObservationStoreDtoV1 {
    VectorF32 {
        values: Vec<f32>,
        offsets: Vec<u64>,
    },
    VectorF64 {
        values: Vec<f64>,
        offsets: Vec<u64>,
    },
    MultivariateF32 {
        values: Vec<f32>,
        series_offsets: Vec<u64>,
        dimension_offsets: Vec<u64>,
    },
    MultivariateF64 {
        values: Vec<f64>,
        series_offsets: Vec<u64>,
        dimension_offsets: Vec<u64>,
    },
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub(crate) enum RetainedTrainingDataDtoV1 {
    None,
    Owned(ObservationStoreDtoV1),
    Reference(ExternalTrainingDataDtoV1),
}
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub(crate) struct ExternalTrainingDataDtoV1 {
    pub(crate) path_utf8: String,
    pub(crate) format: String,
    pub(crate) scalar_kind: String,
    pub(crate) shape: Vec<u64>,
    pub(crate) payload_offset: u64,
    pub(crate) payload_length: u64,
    pub(crate) file_size: u64,
    pub(crate) fingerprint: Option<String>,
}
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub(crate) struct TrainingMetadataDtoV1 {
    pub(crate) sample_count: u64,
    pub(crate) bootstrap: Vec<BootstrapSampleDtoV1>,
}
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub(crate) struct BootstrapSampleDtoV1 {
    pub(crate) sampled_indices: Vec<u64>,
    pub(crate) counts: Vec<u64>,
    pub(crate) out_of_bag: Vec<u64>,
}

/// Framing/DTO error independent of the runtime model error hierarchy.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum SerializationError {
    Truncated,
    InvalidMagic,
    UnsupportedVersion(u32),
    UnknownModelKind(u8),
    KindMismatch {
        header: ModelKindDto,
        body: ModelKindDto,
    },
    LengthOverflow,
    ChecksumMismatch {
        expected: u32,
        actual: u32,
    },
    Encode(String),
    Decode(String),
    Conversion(String),
}
impl fmt::Display for SerializationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl Error for SerializationError {}

fn dto_kind(dto: &ModelDtoV1) -> ModelKindDto {
    match dto {
        ModelDtoV1::Classification(_) => ModelKindDto::Classification,
        ModelDtoV1::Regression(_) => ModelKindDto::Regression,
        ModelDtoV1::Isolation(_) => ModelKindDto::Isolation,
    }
}

/// Encodes a version-one DTO into the portable framed representation.
pub(crate) fn encode_model_v1(dto: &ModelDtoV1) -> Result<Vec<u8>, SerializationError> {
    let body = postcard::to_allocvec(dto)
        .map_err(|error| SerializationError::Encode(error.to_string()))?;
    let length = u64::try_from(body.len()).map_err(|_| SerializationError::LengthOverflow)?;
    let checksum = crc32fast::hash(&body);
    let mut bytes = Vec::with_capacity(HEADER_LEN + body.len());
    bytes.extend_from_slice(&MODEL_MAGIC);
    bytes.extend_from_slice(&MODEL_FORMAT_VERSION.to_le_bytes());
    bytes.push(dto_kind(dto) as u8);
    bytes.extend_from_slice(&length.to_le_bytes());
    bytes.extend_from_slice(&checksum.to_le_bytes());
    bytes.extend_from_slice(&body);
    Ok(bytes)
}

/// Validates framing, size, checksum, version, and model-kind agreement before
/// decoding the version-one DTO body.
pub(crate) fn decode_model_v1(bytes: &[u8]) -> Result<ModelDtoV1, SerializationError> {
    if bytes.len() < HEADER_LEN {
        return Err(SerializationError::Truncated);
    }
    if bytes[..8] != MODEL_MAGIC {
        return Err(SerializationError::InvalidMagic);
    }
    let version = u32::from_le_bytes(bytes[8..12].try_into().expect("fixed slice"));
    if version != MODEL_FORMAT_VERSION {
        return Err(SerializationError::UnsupportedVersion(version));
    }
    let header_kind = ModelKindDto::try_from(bytes[12])?;
    let body_len = u64::from_le_bytes(bytes[13..21].try_into().expect("fixed slice"));
    let body_len = usize::try_from(body_len).map_err(|_| SerializationError::LengthOverflow)?;
    let expected = u32::from_le_bytes(bytes[21..25].try_into().expect("fixed slice"));
    let end = HEADER_LEN
        .checked_add(body_len)
        .ok_or(SerializationError::LengthOverflow)?;
    if end != bytes.len() {
        return Err(SerializationError::Truncated);
    }
    let body = &bytes[HEADER_LEN..end];
    let actual = crc32fast::hash(body);
    if expected != actual {
        return Err(SerializationError::ChecksumMismatch { expected, actual });
    }
    let dto: ModelDtoV1 = postcard::from_bytes(body)
        .map_err(|error| SerializationError::Decode(error.to_string()))?;
    let body_kind = dto_kind(&dto);
    if header_kind != body_kind {
        return Err(SerializationError::KindMismatch {
            header: header_kind,
            body: body_kind,
        });
    }
    Ok(dto)
}

fn usize_to_u64(value: usize) -> Result<u64, SerializationError> {
    u64::try_from(value).map_err(|_| SerializationError::LengthOverflow)
}

fn u64_to_usize(value: u64, field: &str) -> Result<usize, SerializationError> {
    usize::try_from(value)
        .map_err(|_| SerializationError::Conversion(format!("{field} does not fit in usize")))
}

fn usize_slice_to_u64(values: &[usize]) -> Result<Vec<u64>, SerializationError> {
    values.iter().copied().map(usize_to_u64).collect()
}

fn u64_vec_to_usize(values: Vec<u64>, field: &str) -> Result<Vec<usize>, SerializationError> {
    values
        .into_iter()
        .map(|value| u64_to_usize(value, field))
        .collect()
}

impl TryFrom<&super::BootstrapSample> for BootstrapSampleDtoV1 {
    type Error = SerializationError;

    fn try_from(sample: &super::BootstrapSample) -> Result<Self, Self::Error> {
        Ok(Self {
            sampled_indices: usize_slice_to_u64(sample.sampled_positions())?,
            counts: usize_slice_to_u64(sample.multiplicities())?,
            out_of_bag: usize_slice_to_u64(sample.out_of_bag_positions())?,
        })
    }
}

impl TryFrom<BootstrapSampleDtoV1> for super::BootstrapSample {
    type Error = SerializationError;

    fn try_from(dto: BootstrapSampleDtoV1) -> Result<Self, Self::Error> {
        let sampled_positions = u64_vec_to_usize(dto.sampled_indices, "bootstrap draw")?;
        let multiplicities = u64_vec_to_usize(dto.counts, "bootstrap multiplicity")?;
        let out_of_bag_positions = u64_vec_to_usize(dto.out_of_bag, "out-of-bag index")?;
        super::BootstrapSample::from_parts(sampled_positions, multiplicities, out_of_bag_positions)
            .map_err(|error| {
                SerializationError::Conversion(format!("invalid bootstrap metadata: {error}"))
            })
    }
}

impl TryFrom<&super::TrainingMetadata> for TrainingMetadataDtoV1 {
    type Error = SerializationError;

    fn try_from(metadata: &super::TrainingMetadata) -> Result<Self, Self::Error> {
        Ok(Self {
            sample_count: usize_to_u64(metadata.sample_count())?,
            bootstrap: metadata
                .bootstrap()
                .iter()
                .map(BootstrapSampleDtoV1::try_from)
                .collect::<Result<_, _>>()?,
        })
    }
}

impl TryFrom<TrainingMetadataDtoV1> for super::TrainingMetadata {
    type Error = SerializationError;

    fn try_from(dto: TrainingMetadataDtoV1) -> Result<Self, Self::Error> {
        let sample_count = u64_to_usize(dto.sample_count, "training sample count")?;
        if sample_count == 0 {
            return Err(SerializationError::Conversion(
                "training sample count must be positive".to_owned(),
            ));
        }
        let bootstrap = dto
            .bootstrap
            .into_iter()
            .map(super::BootstrapSample::try_from)
            .collect::<Result<Vec<_>, _>>()?;
        if bootstrap
            .iter()
            .any(|sample| sample.multiplicities().len() != sample_count)
        {
            return Err(SerializationError::Conversion(
                "bootstrap metadata width does not match training sample count".to_owned(),
            ));
        }
        Ok(super::TrainingMetadata::new(sample_count, bootstrap))
    }
}

impl TryFrom<&super::ExternalTrainingData> for ExternalTrainingDataDtoV1 {
    type Error = SerializationError;

    fn try_from(descriptor: &super::ExternalTrainingData) -> Result<Self, Self::Error> {
        let path_utf8 = descriptor.path().to_str().ok_or_else(|| {
            SerializationError::Conversion(
                "external training-data path is not valid UTF-8".to_owned(),
            )
        })?;
        Ok(Self {
            path_utf8: path_utf8.to_owned(),
            format: match descriptor.format() {
                super::ExternalDataFormat::Npy => "npy".to_owned(),
            },
            scalar_kind: descriptor.scalar_kind().as_str().to_owned(),
            shape: usize_slice_to_u64(descriptor.shape())?,
            payload_offset: descriptor.payload_offset(),
            payload_length: descriptor.payload_length(),
            file_size: descriptor.file_size(),
            fingerprint: descriptor.fingerprint().map(str::to_owned),
        })
    }
}

impl TryFrom<ExternalTrainingDataDtoV1> for super::ExternalTrainingData {
    type Error = SerializationError;

    fn try_from(dto: ExternalTrainingDataDtoV1) -> Result<Self, Self::Error> {
        if dto.path_utf8.is_empty() {
            return Err(SerializationError::Conversion(
                "external training-data path must not be empty".to_owned(),
            ));
        }
        let format = match dto.format.as_str() {
            "npy" => super::ExternalDataFormat::Npy,
            value => {
                return Err(SerializationError::Conversion(format!(
                    "unsupported external training-data format `{value}`"
                )));
            }
        };
        let scalar_kind = crate::ScalarKind::from_str(&dto.scalar_kind).map_err(|error| {
            SerializationError::Conversion(format!(
                "invalid external training-data scalar kind: {error}"
            ))
        })?;
        let shape = u64_vec_to_usize(dto.shape, "external training-data shape")?;
        if shape.is_empty() || shape.contains(&0) {
            return Err(SerializationError::Conversion(
                "external training-data shape must contain only positive dimensions".to_owned(),
            ));
        }
        let payload_end = dto
            .payload_offset
            .checked_add(dto.payload_length)
            .ok_or_else(|| {
                SerializationError::Conversion(
                    "external training-data payload range overflows u64".to_owned(),
                )
            })?;
        if payload_end > dto.file_size {
            return Err(SerializationError::Conversion(
                "external training-data payload exceeds the recorded file size".to_owned(),
            ));
        }
        if dto.payload_length == 0 {
            return Err(SerializationError::Conversion(
                "external training-data payload must not be empty".to_owned(),
            ));
        }
        Ok(super::ExternalTrainingData::new(
            PathBuf::from(dto.path_utf8),
            format,
            scalar_kind,
            shape.into_boxed_slice(),
            dto.payload_offset,
            dto.payload_length,
            dto.file_size,
            dto.fingerprint.map(String::into_boxed_str),
        ))
    }
}

fn validate_offsets(
    offsets: &[usize],
    value_count: usize,
    name: &str,
) -> Result<(), SerializationError> {
    if offsets.len() < 2 {
        return Err(SerializationError::Conversion(format!(
            "{name} must contain an initial and terminal offset"
        )));
    }
    if offsets[0] != 0 {
        return Err(SerializationError::Conversion(format!(
            "{name} must start at zero"
        )));
    }
    if offsets.last().copied() != Some(value_count) {
        return Err(SerializationError::Conversion(format!(
            "{name} terminal offset must equal the value count"
        )));
    }
    if offsets.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(SerializationError::Conversion(format!(
            "{name} must be strictly increasing; empty observations are unsupported"
        )));
    }
    Ok(())
}

fn flatten_vectors<T: Clone>(rows: &[Vec<T>]) -> Result<(Vec<T>, Vec<u64>), SerializationError> {
    if rows.is_empty() {
        return Err(SerializationError::Conversion(
            "an owned vector store requires at least one observation".to_owned(),
        ));
    }
    let total = rows.iter().try_fold(0usize, |total, row| {
        if row.is_empty() {
            return Err(SerializationError::Conversion(
                "owned vector observations must not be empty".to_owned(),
            ));
        }
        total
            .checked_add(row.len())
            .ok_or(SerializationError::LengthOverflow)
    })?;
    let mut values = Vec::with_capacity(total);
    let mut offsets = Vec::with_capacity(rows.len() + 1);
    offsets.push(0);
    for row in rows {
        values.extend_from_slice(row);
        offsets.push(usize_to_u64(values.len())?);
    }
    Ok((values, offsets))
}

fn rebuild_vectors<T: Clone>(
    values: Vec<T>,
    offsets: Vec<u64>,
) -> Result<Vec<Vec<T>>, SerializationError> {
    let offsets = u64_vec_to_usize(offsets, "vector offset")?;
    validate_offsets(&offsets, values.len(), "vector offsets")?;
    Ok(offsets
        .windows(2)
        .map(|pair| values[pair[0]..pair[1]].to_vec())
        .collect())
}

fn flatten_series<T: Clone>(
    series: &[super::OwnedSeries<T>],
) -> Result<(Vec<T>, Vec<u64>, Vec<u64>), SerializationError> {
    let first = series.first().ok_or_else(|| {
        SerializationError::Conversion(
            "an owned multivariate store requires at least one observation".to_owned(),
        )
    })?;
    let dimension_count = first.dimension_count();
    if dimension_count == 0 {
        return Err(SerializationError::Conversion(
            "multivariate dimension count must be positive".to_owned(),
        ));
    }
    let mut values = Vec::new();
    let mut series_offsets = vec![0u64];
    let mut dimension_offsets = vec![0u64];
    for observation in series {
        if observation.dimension_count() != dimension_count || observation.time_count() == 0 {
            return Err(SerializationError::Conversion(
                "multivariate observations require one positive common dimension count".to_owned(),
            ));
        }
        values.extend_from_slice(observation.values());
        for _ in 0..dimension_count {
            let next = dimension_offsets
                .last()
                .copied()
                .and_then(|offset| offset.checked_add(usize_to_u64(observation.time_count()).ok()?))
                .ok_or(SerializationError::LengthOverflow)?;
            dimension_offsets.push(next);
        }
        series_offsets.push(usize_to_u64(values.len())?);
    }
    Ok((values, series_offsets, dimension_offsets))
}

fn rebuild_series<T: Clone>(
    values: Vec<T>,
    series_offsets: Vec<u64>,
    dimension_offsets: Vec<u64>,
) -> Result<Vec<super::OwnedSeries<T>>, SerializationError> {
    let series_offsets = u64_vec_to_usize(series_offsets, "series offset")?;
    let dimension_offsets = u64_vec_to_usize(dimension_offsets, "dimension offset")?;
    validate_offsets(&series_offsets, values.len(), "series offsets")?;
    validate_offsets(&dimension_offsets, values.len(), "dimension offsets")?;
    let observation_count = series_offsets.len() - 1;
    let dimension_segment_count = dimension_offsets.len() - 1;
    if dimension_segment_count % observation_count != 0 {
        return Err(SerializationError::Conversion(
            "dimension offsets do not divide evenly across observations".to_owned(),
        ));
    }
    let dimension_count = dimension_segment_count / observation_count;
    if dimension_count == 0 {
        return Err(SerializationError::Conversion(
            "multivariate dimension count must be positive".to_owned(),
        ));
    }
    let mut rebuilt = Vec::with_capacity(observation_count);
    for observation_index in 0..observation_count {
        let series_start = series_offsets[observation_index];
        let series_end = series_offsets[observation_index + 1];
        let first_dimension = observation_index * dimension_count;
        if dimension_offsets[first_dimension] != series_start
            || dimension_offsets[first_dimension + dimension_count] != series_end
        {
            return Err(SerializationError::Conversion(
                "series and dimension offsets disagree".to_owned(),
            ));
        }
        let dimension_lengths =
            &dimension_offsets[first_dimension..=first_dimension + dimension_count];
        let time_count = dimension_lengths[1] - dimension_lengths[0];
        if time_count == 0
            || dimension_lengths
                .windows(2)
                .any(|pair| pair[1] - pair[0] != time_count)
        {
            return Err(SerializationError::Conversion(
                "each multivariate observation must be rectangular".to_owned(),
            ));
        }
        rebuilt.push(
            super::OwnedSeries::from_parts(
                values[series_start..series_end].to_vec(),
                dimension_count,
                time_count,
            )
            .map_err(|error| {
                SerializationError::Conversion(format!("invalid multivariate observation: {error}"))
            })?,
        );
    }
    Ok(rebuilt)
}

impl TryFrom<&super::OwnedObservationStore> for ObservationStoreDtoV1 {
    type Error = SerializationError;

    fn try_from(store: &super::OwnedObservationStore) -> Result<Self, Self::Error> {
        match store {
            super::OwnedObservationStore::Vector(super::OwnedVectorStore::F32(rows)) => {
                let (values, offsets) = flatten_vectors(rows)?;
                Ok(Self::VectorF32 { values, offsets })
            }
            super::OwnedObservationStore::Vector(super::OwnedVectorStore::F64(rows)) => {
                let (values, offsets) = flatten_vectors(rows)?;
                Ok(Self::VectorF64 { values, offsets })
            }
            super::OwnedObservationStore::Multivariate(super::OwnedMultivariateStore::F32(
                series,
            )) => {
                let (values, series_offsets, dimension_offsets) = flatten_series(series)?;
                Ok(Self::MultivariateF32 {
                    values,
                    series_offsets,
                    dimension_offsets,
                })
            }
            super::OwnedObservationStore::Multivariate(super::OwnedMultivariateStore::F64(
                series,
            )) => {
                let (values, series_offsets, dimension_offsets) = flatten_series(series)?;
                Ok(Self::MultivariateF64 {
                    values,
                    series_offsets,
                    dimension_offsets,
                })
            }
        }
    }
}

impl TryFrom<ObservationStoreDtoV1> for super::OwnedObservationStore {
    type Error = SerializationError;

    fn try_from(dto: ObservationStoreDtoV1) -> Result<Self, Self::Error> {
        Ok(match dto {
            ObservationStoreDtoV1::VectorF32 { values, offsets } => Self::Vector(
                super::OwnedVectorStore::F32(rebuild_vectors(values, offsets)?),
            ),
            ObservationStoreDtoV1::VectorF64 { values, offsets } => Self::Vector(
                super::OwnedVectorStore::F64(rebuild_vectors(values, offsets)?),
            ),
            ObservationStoreDtoV1::MultivariateF32 {
                values,
                series_offsets,
                dimension_offsets,
            } => Self::Multivariate(super::OwnedMultivariateStore::F32(rebuild_series(
                values,
                series_offsets,
                dimension_offsets,
            )?)),
            ObservationStoreDtoV1::MultivariateF64 {
                values,
                series_offsets,
                dimension_offsets,
            } => Self::Multivariate(super::OwnedMultivariateStore::F64(rebuild_series(
                values,
                series_offsets,
                dimension_offsets,
            )?)),
        })
    }
}

impl TryFrom<&super::RetainedTrainingData> for RetainedTrainingDataDtoV1 {
    type Error = SerializationError;

    fn try_from(retained: &super::RetainedTrainingData) -> Result<Self, Self::Error> {
        match retained {
            super::RetainedTrainingData::None => Ok(Self::None),
            super::RetainedTrainingData::Owned(store) => {
                Ok(Self::Owned(ObservationStoreDtoV1::try_from(store)?))
            }
            super::RetainedTrainingData::Reference(descriptor) => Ok(Self::Reference(
                ExternalTrainingDataDtoV1::try_from(descriptor)?,
            )),
        }
    }
}

impl TryFrom<RetainedTrainingDataDtoV1> for super::RetainedTrainingData {
    type Error = SerializationError;

    fn try_from(dto: RetainedTrainingDataDtoV1) -> Result<Self, Self::Error> {
        match dto {
            RetainedTrainingDataDtoV1::None => Ok(Self::None),
            RetainedTrainingDataDtoV1::Owned(store) => {
                Ok(Self::Owned(super::OwnedObservationStore::try_from(store)?))
            }
            RetainedTrainingDataDtoV1::Reference(descriptor) => Ok(Self::Reference(
                super::ExternalTrainingData::try_from(descriptor)?,
            )),
        }
    }
}

fn empty_distance_dto(family: &str) -> DistanceConfigDtoV1 {
    DistanceConfigDtoV1 {
        family: family.to_owned(),
        floats: Vec::new(),
        integers: Vec::new(),
        flags: Vec::new(),
        strings: Vec::new(),
    }
}

fn push_window(
    dto: &mut DistanceConfigDtoV1,
    window: Option<usize>,
) -> Result<(), SerializationError> {
    if let Some(window) = window {
        dto.integers
            .push(("window".to_owned(), usize_to_u64(window)?));
    }
    Ok(())
}

fn push_float(dto: &mut DistanceConfigDtoV1, name: &str, value: crate::distance::FiniteF64) {
    dto.floats.push((name.to_owned(), value.get()));
}

fn no_overlap_name(policy: crate::distance::NoOverlapPolicy) -> &'static str {
    match policy {
        crate::distance::NoOverlapPolicy::Infinite => "infinite",
        crate::distance::NoOverlapPolicy::Zero => "zero",
    }
}

fn parse_no_overlap(value: &str) -> Result<crate::distance::NoOverlapPolicy, SerializationError> {
    match value {
        "infinite" => Ok(crate::distance::NoOverlapPolicy::Infinite),
        "zero" => Ok(crate::distance::NoOverlapPolicy::Zero),
        other => Err(SerializationError::Conversion(format!(
            "unknown no-overlap policy `{other}`"
        ))),
    }
}

struct DistanceParameters {
    floats: std::collections::BTreeMap<String, f64>,
    integers: std::collections::BTreeMap<String, u64>,
    flags: std::collections::BTreeMap<String, bool>,
    strings: std::collections::BTreeMap<String, String>,
}

impl DistanceParameters {
    fn from_dto(dto: DistanceConfigDtoV1) -> Result<(String, Self), SerializationError> {
        fn collect<T>(
            entries: Vec<(String, T)>,
            kind: &str,
        ) -> Result<std::collections::BTreeMap<String, T>, SerializationError> {
            let mut result = std::collections::BTreeMap::new();
            for (name, value) in entries {
                if result.insert(name.clone(), value).is_some() {
                    return Err(SerializationError::Conversion(format!(
                        "duplicate {kind} distance parameter `{name}`"
                    )));
                }
            }
            Ok(result)
        }
        Ok((
            dto.family,
            Self {
                floats: collect(dto.floats, "floating-point")?,
                integers: collect(dto.integers, "integer")?,
                flags: collect(dto.flags, "boolean")?,
                strings: collect(dto.strings, "string")?,
            },
        ))
    }

    fn finite(&mut self, name: &str) -> Result<crate::distance::FiniteF64, SerializationError> {
        let value = self.floats.remove(name).ok_or_else(|| {
            SerializationError::Conversion(format!(
                "missing floating-point distance parameter `{name}`"
            ))
        })?;
        crate::distance::FiniteF64::new(value).map_err(|error| {
            SerializationError::Conversion(format!("invalid distance parameter `{name}`: {error}"))
        })
    }

    fn window(&mut self) -> Result<Option<usize>, SerializationError> {
        self.integers
            .remove("window")
            .map(|value| u64_to_usize(value, "distance window"))
            .transpose()
    }

    fn string(&mut self, name: &str) -> Result<String, SerializationError> {
        self.strings.remove(name).ok_or_else(|| {
            SerializationError::Conversion(format!("missing string distance parameter `{name}`"))
        })
    }

    fn finish(self, family: &str) -> Result<(), SerializationError> {
        if self.floats.is_empty()
            && self.integers.is_empty()
            && self.flags.is_empty()
            && self.strings.is_empty()
        {
            Ok(())
        } else {
            Err(SerializationError::Conversion(format!(
                "distance family `{family}` contains unexpected parameters"
            )))
        }
    }
}

impl TryFrom<&crate::distance::DistanceConfig> for DistanceConfigDtoV1 {
    type Error = SerializationError;

    fn try_from(config: &crate::distance::DistanceConfig) -> Result<Self, Self::Error> {
        use crate::distance::DistanceConfig as C;
        #[allow(unreachable_patterns)]
        let mut dto = match config {
            C::SquaredEuclidean => empty_distance_dto("squared_euclidean"),
            C::Manhattan => empty_distance_dto("manhattan"),
            C::Cosine => empty_distance_dto("cosine"),
            C::Cid => empty_distance_dto("cid"),
            C::Sbd => empty_distance_dto("sbd"),
            C::SquaredEuclideanIndependent => empty_distance_dto("squared_euclidean_independent"),
            C::ManhattanIndependent => empty_distance_dto("manhattan_independent"),
            C::CidIndependent => empty_distance_dto("cid_independent"),
            C::SbdIndependent => empty_distance_dto("sbd_independent"),
            C::Dtw(p)
            | C::Ddtw(p)
            | C::DtwIndependent(p)
            | C::DdtwIndependent(p)
            | C::DtwDependent(p)
            | C::DdtwDependent(p) => {
                let family = match config.kind() {
                    crate::distance::DistanceKind::Dtw => "dtw",
                    crate::distance::DistanceKind::Ddtw => "ddtw",
                    crate::distance::DistanceKind::DtwIndependent => "dtw_independent",
                    crate::distance::DistanceKind::DdtwIndependent => "ddtw_independent",
                    crate::distance::DistanceKind::DtwDependent => "dtw_dependent",
                    crate::distance::DistanceKind::DdtwDependent => "ddtw_dependent",
                    _ => unreachable!(),
                };
                let mut dto = empty_distance_dto(family);
                push_window(&mut dto, p.window)?;
                dto
            }
            C::Wdtw(p)
            | C::Wddtw(p)
            | C::WdtwIndependent(p)
            | C::WddtwIndependent(p)
            | C::WdtwDependent(p)
            | C::WddtwDependent(p) => {
                let family = match config.kind() {
                    crate::distance::DistanceKind::Wdtw => "wdtw",
                    crate::distance::DistanceKind::Wddtw => "wddtw",
                    crate::distance::DistanceKind::WdtwIndependent => "wdtw_independent",
                    crate::distance::DistanceKind::WddtwIndependent => "wddtw_independent",
                    crate::distance::DistanceKind::WdtwDependent => "wdtw_dependent",
                    crate::distance::DistanceKind::WddtwDependent => "wddtw_dependent",
                    _ => unreachable!(),
                };
                let mut dto = empty_distance_dto(family);
                push_window(&mut dto, p.window)?;
                push_float(&mut dto, "stiffness", p.stiffness);
                dto
            }
            C::Erp(p) | C::ErpIndependent(p) => {
                let mut dto = empty_distance_dto(if matches!(config, C::Erp(_)) {
                    "erp"
                } else {
                    "erp_independent"
                });
                push_window(&mut dto, p.window)?;
                push_float(&mut dto, "gap", p.gap);
                dto
            }
            C::Lcss(p) | C::LcssIndependent(p) => {
                let mut dto = empty_distance_dto(if matches!(config, C::Lcss(_)) {
                    "lcss"
                } else {
                    "lcss_independent"
                });
                push_window(&mut dto, p.window)?;
                push_float(&mut dto, "epsilon", p.epsilon);
                dto
            }
            C::Msm(p) | C::MsmIndependent(p) => {
                let mut dto = empty_distance_dto(if matches!(config, C::Msm(_)) {
                    "msm"
                } else {
                    "msm_independent"
                });
                push_float(&mut dto, "cost", p.cost);
                dto
            }
            C::Twe(p) | C::TweIndependent(p) => {
                let mut dto = empty_distance_dto(if matches!(config, C::Twe(_)) {
                    "twe"
                } else {
                    "twe_independent"
                });
                push_float(&mut dto, "stiffness", p.stiffness);
                push_float(&mut dto, "penalty", p.penalty);
                dto
            }
            C::ShapeHog1dDtw(p) | C::ShapeHog1dDtwIndependent(p) | C::ShapeHogDtw(p) => {
                let family = if matches!(config, C::ShapeHog1dDtw(_)) {
                    "shape_hog_1d_dtw"
                } else if matches!(config, C::ShapeHog1dDtwIndependent(_)) {
                    "shape_hog_1d_dtw_independent"
                } else {
                    "shape_hog_dtw"
                };
                let mut dto = empty_distance_dto(family);
                push_window(&mut dto, p.window)?;
                dto
            }
            C::NaNSquaredEuclidean { no_overlap } => {
                let mut dto = empty_distance_dto("nan_squared_euclidean");
                dto.strings.push((
                    "no_overlap".to_owned(),
                    no_overlap_name(*no_overlap).to_owned(),
                ));
                dto
            }
            C::NaNSquaredEuclideanIndependent { no_overlap } => {
                let mut dto = empty_distance_dto("nan_squared_euclidean_independent");
                dto.strings.push((
                    "no_overlap".to_owned(),
                    no_overlap_name(*no_overlap).to_owned(),
                ));
                dto
            }
            C::DtwArow(p) | C::DtwArowIndependent(p) | C::DtwArowDependent(p) => {
                let family = if matches!(config, C::DtwArow(_)) {
                    "dtw_arow"
                } else if matches!(config, C::DtwArowIndependent(_)) {
                    "dtw_arow_independent"
                } else {
                    "dtw_arow_dependent"
                };
                let mut dto = empty_distance_dto(family);
                push_window(&mut dto, p.window)?;
                dto
            }
            _ => {
                return Err(SerializationError::Conversion(
                    "unsupported future distance configuration".to_owned(),
                ));
            }
        };
        dto.floats.sort_by(|a, b| a.0.cmp(&b.0));
        dto.integers.sort_by(|a, b| a.0.cmp(&b.0));
        dto.flags.sort_by(|a, b| a.0.cmp(&b.0));
        dto.strings.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(dto)
    }
}

impl TryFrom<DistanceConfigDtoV1> for crate::distance::DistanceConfig {
    type Error = SerializationError;
    fn try_from(dto: DistanceConfigDtoV1) -> Result<Self, Self::Error> {
        use crate::distance::{
            DistanceConfig as C, DtwArowParameters as A, DtwParameters as D, ErpParameters as E,
            LcssParameters as L, MsmParameters as M, ShapeDtwParameters as S, TweParameters as T,
            WeightedDtwParameters as W,
        };
        let (family, mut p) = DistanceParameters::from_dto(dto)?;
        let config = match family.as_str() {
            "squared_euclidean" => C::SquaredEuclidean,
            "manhattan" => C::Manhattan,
            "cosine" => C::Cosine,
            "cid" => C::Cid,
            "sbd" => C::Sbd,
            "squared_euclidean_independent" => C::SquaredEuclideanIndependent,
            "manhattan_independent" => C::ManhattanIndependent,
            "cid_independent" => C::CidIndependent,
            "sbd_independent" => C::SbdIndependent,
            "dtw" => C::Dtw(D {
                window: p.window()?,
            }),
            "ddtw" => C::Ddtw(D {
                window: p.window()?,
            }),
            "dtw_independent" => C::DtwIndependent(D {
                window: p.window()?,
            }),
            "ddtw_independent" => C::DdtwIndependent(D {
                window: p.window()?,
            }),
            "dtw_dependent" => C::DtwDependent(D {
                window: p.window()?,
            }),
            "ddtw_dependent" => C::DdtwDependent(D {
                window: p.window()?,
            }),
            "wdtw" => C::Wdtw(W {
                window: p.window()?,
                stiffness: p.finite("stiffness")?,
            }),
            "wddtw" => C::Wddtw(W {
                window: p.window()?,
                stiffness: p.finite("stiffness")?,
            }),
            "wdtw_independent" => C::WdtwIndependent(W {
                window: p.window()?,
                stiffness: p.finite("stiffness")?,
            }),
            "wddtw_independent" => C::WddtwIndependent(W {
                window: p.window()?,
                stiffness: p.finite("stiffness")?,
            }),
            "wdtw_dependent" => C::WdtwDependent(W {
                window: p.window()?,
                stiffness: p.finite("stiffness")?,
            }),
            "wddtw_dependent" => C::WddtwDependent(W {
                window: p.window()?,
                stiffness: p.finite("stiffness")?,
            }),
            "erp" => C::Erp(E {
                window: p.window()?,
                gap: p.finite("gap")?,
            }),
            "erp_independent" => C::ErpIndependent(E {
                window: p.window()?,
                gap: p.finite("gap")?,
            }),
            "lcss" => C::Lcss(L {
                window: p.window()?,
                epsilon: p.finite("epsilon")?,
            }),
            "lcss_independent" => C::LcssIndependent(L {
                window: p.window()?,
                epsilon: p.finite("epsilon")?,
            }),
            "msm" => C::Msm(M {
                cost: p.finite("cost")?,
            }),
            "msm_independent" => C::MsmIndependent(M {
                cost: p.finite("cost")?,
            }),
            "twe" => C::Twe(T {
                stiffness: p.finite("stiffness")?,
                penalty: p.finite("penalty")?,
            }),
            "twe_independent" => C::TweIndependent(T {
                stiffness: p.finite("stiffness")?,
                penalty: p.finite("penalty")?,
            }),
            "shape_hog_1d_dtw" => C::ShapeHog1dDtw(S {
                window: p.window()?,
            }),
            "shape_hog_1d_dtw_independent" => C::ShapeHog1dDtwIndependent(S {
                window: p.window()?,
            }),
            "shape_hog_dtw" => C::ShapeHogDtw(S {
                window: p.window()?,
            }),
            "nan_squared_euclidean" => C::NaNSquaredEuclidean {
                no_overlap: parse_no_overlap(&p.string("no_overlap")?)?,
            },
            "nan_squared_euclidean_independent" => C::NaNSquaredEuclideanIndependent {
                no_overlap: parse_no_overlap(&p.string("no_overlap")?)?,
            },
            "dtw_arow" => C::DtwArow(A {
                window: p.window()?,
            }),
            "dtw_arow_independent" => C::DtwArowIndependent(A {
                window: p.window()?,
            }),
            "dtw_arow_dependent" => C::DtwArowDependent(A {
                window: p.window()?,
            }),
            other => {
                return Err(SerializationError::Conversion(format!(
                    "unknown distance family `{other}`"
                )));
            }
        };
        p.finish(&family)?;
        Ok(config)
    }
}

impl TryFrom<&super::LeafPayload> for LeafPayloadDtoV1 {
    type Error = SerializationError;

    fn try_from(payload: &super::LeafPayload) -> Result<Self, Self::Error> {
        match payload {
            super::LeafPayload::Empty => Ok(Self::Empty),
            super::LeafPayload::Classification(leaf) => {
                let counts = leaf
                    .all_counts()
                    .iter()
                    .map(|output| {
                        output
                            .iter()
                            .map(|&(class_id, count)| Ok((class_id, usize_to_u64(count)?)))
                            .collect::<Result<Vec<_>, SerializationError>>()
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Self::Classification { counts })
            }
            super::LeafPayload::Regression(leaf) => {
                if leaf.means().iter().any(|value| !value.is_finite()) {
                    return Err(SerializationError::Conversion(
                        "regression leaf contains a non-finite mean".to_owned(),
                    ));
                }
                Ok(Self::Regression {
                    means: leaf.means().to_vec(),
                })
            }
            super::LeafPayload::Isolation { occurrence_count } => {
                if *occurrence_count == 0 {
                    return Err(SerializationError::Conversion(
                        "isolation leaf occurrence count must be positive".to_owned(),
                    ));
                }
                Ok(Self::Isolation {
                    occurrence_count: usize_to_u64(*occurrence_count)?,
                })
            }
        }
    }
}

impl TryFrom<LeafPayloadDtoV1> for super::LeafPayload {
    type Error = SerializationError;

    fn try_from(dto: LeafPayloadDtoV1) -> Result<Self, Self::Error> {
        match dto {
            LeafPayloadDtoV1::Empty => Ok(Self::Empty),
            LeafPayloadDtoV1::Classification { counts } => {
                let counts = counts
                    .into_iter()
                    .map(|output| {
                        output
                            .into_iter()
                            .map(|(class_id, count)| {
                                Ok((class_id, u64_to_usize(count, "classification class count")?))
                            })
                            .collect::<Result<Vec<_>, SerializationError>>()
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let leaf = super::ClassificationLeaf::from_counts(counts).map_err(|error| {
                    SerializationError::Conversion(format!(
                        "invalid classification leaf payload: {error}"
                    ))
                })?;
                Ok(Self::Classification(leaf))
            }
            LeafPayloadDtoV1::Regression { means } => {
                let leaf = super::RegressionLeaf::from_means(means).map_err(|error| {
                    SerializationError::Conversion(format!(
                        "invalid regression leaf payload: {error}"
                    ))
                })?;
                Ok(Self::Regression(leaf))
            }
            LeafPayloadDtoV1::Isolation { occurrence_count } => {
                let occurrence_count =
                    u64_to_usize(occurrence_count, "isolation occurrence count")?;
                if occurrence_count == 0 {
                    return Err(SerializationError::Conversion(
                        "isolation leaf occurrence count must be positive".to_owned(),
                    ));
                }
                Ok(Self::Isolation { occurrence_count })
            }
        }
    }
}

pub(crate) fn validate_leaf_payload_for_model(
    payload: &super::LeafPayload,
    model_kind: ModelKindDto,
    expected_output_count: Option<usize>,
) -> Result<(), SerializationError> {
    match (model_kind, payload) {
        (ModelKindDto::Classification, super::LeafPayload::Classification(leaf)) => {
            if expected_output_count != Some(leaf.output_count()) {
                return Err(SerializationError::Conversion(
                    "classification leaf output count does not match the fitted model".to_owned(),
                ));
            }
            Ok(())
        }
        (ModelKindDto::Regression, super::LeafPayload::Regression(leaf)) => {
            if expected_output_count != Some(leaf.means().len()) {
                return Err(SerializationError::Conversion(
                    "regression leaf output count does not match the fitted model".to_owned(),
                ));
            }
            Ok(())
        }
        (ModelKindDto::Isolation, super::LeafPayload::Isolation { occurrence_count })
            if *occurrence_count > 0 =>
        {
            Ok(())
        }
        (_, super::LeafPayload::Empty) => Err(SerializationError::Conversion(
            "a fitted model must not contain an empty leaf payload".to_owned(),
        )),
        _ => Err(SerializationError::Conversion(
            "leaf payload kind does not match the fitted model kind".to_owned(),
        )),
    }
}

fn exemplar_count(store: &super::OwnedObservationStore) -> usize {
    match store {
        super::OwnedObservationStore::Vector(store) => store.len(),
        super::OwnedObservationStore::Multivariate(store) => store.len(),
    }
}

fn validate_tree_graph(
    nodes: &[super::TreeNode],
    root: usize,
    exemplar_count: usize,
    model_kind: ModelKindDto,
    expected_output_count: Option<usize>,
) -> Result<(), SerializationError> {
    if nodes.is_empty() {
        return Err(SerializationError::Conversion(
            "a fitted tree requires at least one node".to_owned(),
        ));
    }
    if root >= nodes.len() {
        return Err(SerializationError::Conversion(
            "tree root is outside the node arena".to_owned(),
        ));
    }
    if exemplar_count == 0
        && nodes
            .iter()
            .any(|node| matches!(node, super::TreeNode::Split(_)))
    {
        return Err(SerializationError::Conversion(
            "a split tree requires at least one stored exemplar".to_owned(),
        ));
    }

    let mut parent_counts = vec![0usize; nodes.len()];
    for (node_index, node) in nodes.iter().enumerate() {
        match node {
            super::TreeNode::Leaf(leaf) => {
                if leaf.occurrence_count() == 0
                    || leaf.distinct_count() == 0
                    || leaf.distinct_count() > leaf.occurrence_count()
                {
                    return Err(SerializationError::Conversion(format!(
                        "leaf node {node_index} has invalid occurrence/distinct counts"
                    )));
                }
                validate_leaf_payload_for_model(leaf.payload(), model_kind, expected_output_count)?;
            }
            super::TreeNode::Split(split) => {
                if split.branches().is_empty() {
                    return Err(SerializationError::Conversion(format!(
                        "split node {node_index} has no branches"
                    )));
                }
                if split
                    .selected_dimensions()
                    .windows(2)
                    .any(|pair| pair[0] >= pair[1])
                {
                    return Err(SerializationError::Conversion(format!(
                        "split node {node_index} dimensions must be strictly increasing"
                    )));
                }
                for branch in split.branches() {
                    if branch.exemplar().index() >= exemplar_count {
                        return Err(SerializationError::Conversion(format!(
                            "split node {node_index} references an unavailable exemplar"
                        )));
                    }
                    let child = branch.child().get();
                    if child >= nodes.len() {
                        return Err(SerializationError::Conversion(format!(
                            "split node {node_index} references a child outside the node arena"
                        )));
                    }
                    if child == node_index {
                        return Err(SerializationError::Conversion(format!(
                            "split node {node_index} contains a self-loop"
                        )));
                    }
                    parent_counts[child] = parent_counts[child]
                        .checked_add(1)
                        .ok_or(SerializationError::LengthOverflow)?;
                }
            }
        }
    }
    if parent_counts[root] != 0 {
        return Err(SerializationError::Conversion(
            "tree root must not have a parent".to_owned(),
        ));
    }
    if parent_counts
        .iter()
        .enumerate()
        .any(|(index, &count)| index != root && count != 1)
    {
        return Err(SerializationError::Conversion(
            "every non-root tree node must have exactly one parent".to_owned(),
        ));
    }

    let mut colors = vec![0u8; nodes.len()];
    let mut stack = vec![(root, false)];
    while let Some((node_index, exiting)) = stack.pop() {
        if exiting {
            colors[node_index] = 2;
            continue;
        }
        match colors[node_index] {
            1 => {
                return Err(SerializationError::Conversion(
                    "tree node graph contains a directed cycle".to_owned(),
                ));
            }
            2 => continue,
            _ => {}
        }
        colors[node_index] = 1;
        stack.push((node_index, true));
        if let super::TreeNode::Split(split) = &nodes[node_index] {
            for branch in split.branches().iter().rev() {
                stack.push((branch.child().get(), false));
            }
        }
    }
    if colors.iter().any(|&color| color != 2) {
        return Err(SerializationError::Conversion(
            "tree contains nodes unreachable from its root".to_owned(),
        ));
    }
    Ok(())
}

pub(crate) fn tree_to_dto(
    tree: &super::ProximityTree,
    model_kind: ModelKindDto,
    expected_output_count: Option<usize>,
) -> Result<TreeDtoV1, SerializationError> {
    let root = tree.root().ok_or_else(|| {
        SerializationError::Conversion("cannot serialize an empty fitted tree".to_owned())
    })?;
    let exemplars = tree.exemplars().ok_or_else(|| {
        SerializationError::Conversion("fitted tree has no exemplar store".to_owned())
    })?;
    validate_tree_graph(
        tree.nodes(),
        root.get(),
        exemplar_count(exemplars),
        model_kind,
        expected_output_count,
    )?;

    let nodes = tree
        .nodes()
        .iter()
        .map(|node| match node {
            super::TreeNode::Split(split) => Ok(TreeNodeDtoV1::Split {
                distance: DistanceConfigDtoV1::try_from(split.distance())?,
                selected_dimensions: usize_slice_to_u64(split.selected_dimensions())?,
                branches: split
                    .branches()
                    .iter()
                    .map(|branch| {
                        Ok(BranchDtoV1 {
                            exemplar_id: branch.exemplar().get(),
                            child: usize_to_u64(branch.child().get())?,
                        })
                    })
                    .collect::<Result<Vec<_>, SerializationError>>()?,
            }),
            super::TreeNode::Leaf(leaf) => Ok(TreeNodeDtoV1::Leaf {
                payload: LeafPayloadDtoV1::try_from(leaf.payload())?,
                occurrence_count: usize_to_u64(leaf.occurrence_count())?,
                distinct_count: usize_to_u64(leaf.distinct_count())?,
            }),
        })
        .collect::<Result<Vec<_>, SerializationError>>()?;

    Ok(TreeDtoV1 {
        root: usize_to_u64(root.get())?,
        nodes,
        exemplars: ObservationStoreDtoV1::try_from(exemplars)?,
    })
}

pub(crate) fn tree_from_dto(
    dto: TreeDtoV1,
    model_kind: ModelKindDto,
    expected_output_count: Option<usize>,
) -> Result<super::ProximityTree, SerializationError> {
    let root = u64_to_usize(dto.root, "tree root")?;
    if root != 0 {
        return Err(SerializationError::Conversion(
            "format version 1 requires the tree root to occupy arena index zero".to_owned(),
        ));
    }
    let exemplars = super::OwnedObservationStore::try_from(dto.exemplars)?;
    let nodes = dto
        .nodes
        .into_iter()
        .map(|node| match node {
            TreeNodeDtoV1::Split {
                distance,
                selected_dimensions,
                branches,
            } => {
                let selected_dimensions =
                    u64_vec_to_usize(selected_dimensions, "selected dimension")?;
                let branches = branches
                    .into_iter()
                    .map(|branch| {
                        let child = u64_to_usize(branch.child, "child node index")?;
                        Ok(super::ExemplarBranch::new(
                            super::ExemplarId::new(branch.exemplar_id),
                            super::NodeId::new(child),
                        ))
                    })
                    .collect::<Result<Vec<_>, SerializationError>>()?;
                Ok(super::TreeNode::Split(super::SplitNode::new(
                    crate::distance::DistanceConfig::try_from(distance)?,
                    selected_dimensions,
                    branches,
                )))
            }
            TreeNodeDtoV1::Leaf {
                payload,
                occurrence_count,
                distinct_count,
            } => {
                let payload = super::LeafPayload::try_from(payload)?;
                let leaf = super::LeafNode::from_parts(
                    u64_to_usize(occurrence_count, "leaf occurrence count")?,
                    u64_to_usize(distinct_count, "leaf distinct count")?,
                    payload,
                )
                .map_err(|error| {
                    SerializationError::Conversion(format!("invalid fitted leaf node: {error}"))
                })?;
                Ok(super::TreeNode::Leaf(leaf))
            }
        })
        .collect::<Result<Vec<_>, SerializationError>>()?;

    validate_tree_graph(
        &nodes,
        root,
        exemplar_count(&exemplars),
        model_kind,
        expected_output_count,
    )?;
    super::ProximityTree::from_parts(nodes, exemplars)
        .map_err(|error| SerializationError::Conversion(format!("invalid fitted tree: {error}")))
}

fn observation_store_len(store: &super::OwnedObservationStore) -> usize {
    match store {
        super::OwnedObservationStore::Vector(store) => store.len(),
        super::OwnedObservationStore::Multivariate(store) => store.len(),
    }
}

fn validate_retention_state(
    config: &super::ForestConfig,
    retained: &super::RetainedTrainingData,
    sample_count: usize,
) -> Result<(), SerializationError> {
    match (config.data_retention, retained) {
        (super::DataRetention::Exemplars, super::RetainedTrainingData::None) => Ok(()),
        (super::DataRetention::Copy, super::RetainedTrainingData::Owned(store)) => {
            if observation_store_len(store) != sample_count {
                return Err(SerializationError::Conversion(
                    "embedded training-data count does not match training metadata".to_owned(),
                ));
            }
            Ok(())
        }
        (super::DataRetention::Reference, super::RetainedTrainingData::Reference(descriptor)) => {
            if descriptor.shape().first().copied() != Some(sample_count) {
                return Err(SerializationError::Conversion(
                    "external training-data shape does not match training metadata".to_owned(),
                ));
            }
            if config.external_training_data.as_ref() != Some(descriptor) {
                return Err(SerializationError::Conversion(
                    "forest configuration and retained external descriptor disagree".to_owned(),
                ));
            }
            Ok(())
        }
        _ => Err(SerializationError::Conversion(
            "retained training data does not match the configured retention policy".to_owned(),
        )),
    }
}

fn forest_to_dto(
    forest: &super::FittedForest,
    model_kind: ModelKindDto,
    expected_output_count: Option<usize>,
) -> Result<ForestDtoV1, SerializationError> {
    let config = forest.config();
    let trees = forest.trees();
    let metadata = forest.training_metadata();
    if trees.is_empty() || trees.len() != config.tree_count.get() {
        return Err(SerializationError::Conversion(
            "fitted tree count does not match forest configuration".to_owned(),
        ));
    }
    if metadata.sample_count() == 0 || metadata.bootstrap().len() != trees.len() {
        return Err(SerializationError::Conversion(
            "training metadata does not match the fitted forest".to_owned(),
        ));
    }
    validate_retention_state(
        config,
        forest.retained_training_data(),
        metadata.sample_count(),
    )?;

    Ok(ForestDtoV1 {
        config: ForestConfigDtoV1::try_from(config)?,
        trees: trees
            .iter()
            .map(|tree| tree_to_dto(tree, model_kind, expected_output_count))
            .collect::<Result<Vec<_>, _>>()?,
        retained_training_data: RetainedTrainingDataDtoV1::try_from(
            forest.retained_training_data(),
        )?,
        training_metadata: TrainingMetadataDtoV1::try_from(metadata)?,
    })
}

fn forest_from_dto(
    dto: ForestDtoV1,
    model_kind: ModelKindDto,
    expected_output_count: Option<usize>,
) -> Result<super::FittedForest, SerializationError> {
    let mut config = super::ForestConfig::try_from(dto.config)?;
    let retained = super::RetainedTrainingData::try_from(dto.retained_training_data)?;
    if let super::RetainedTrainingData::Reference(descriptor) = &retained {
        config.external_training_data = Some(descriptor.clone());
    } else {
        config.external_training_data = None;
    }
    config.validate().map_err(|error| {
        SerializationError::Conversion(format!("invalid restored forest configuration: {error}"))
    })?;

    let metadata = super::TrainingMetadata::try_from(dto.training_metadata)?;
    if dto.trees.len() != config.tree_count.get() || metadata.bootstrap().len() != dto.trees.len() {
        return Err(SerializationError::Conversion(
            "serialized tree and metadata counts do not match forest configuration".to_owned(),
        ));
    }
    validate_retention_state(&config, &retained, metadata.sample_count())?;
    let trees = dto
        .trees
        .into_iter()
        .map(|tree| tree_from_dto(tree, model_kind, expected_output_count))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(super::FittedForest::new(config, trees, retained, metadata))
}

impl TryFrom<&super::ClassificationForest> for ClassificationModelDtoV1 {
    type Error = SerializationError;

    fn try_from(model: &super::ClassificationForest) -> Result<Self, Self::Error> {
        if model.output_count() == 0 {
            return Err(SerializationError::Conversion(
                "classification output count must be positive".to_owned(),
            ));
        }
        Ok(Self {
            forest: forest_to_dto(
                model.forest(),
                ModelKindDto::Classification,
                Some(model.output_count()),
            )?,
            output_count: u32::try_from(model.output_count())
                .map_err(|_| SerializationError::LengthOverflow)?,
        })
    }
}

impl TryFrom<ClassificationModelDtoV1> for super::ClassificationForest {
    type Error = SerializationError;

    fn try_from(dto: ClassificationModelDtoV1) -> Result<Self, Self::Error> {
        let output_count =
            usize::try_from(dto.output_count).map_err(|_| SerializationError::LengthOverflow)?;
        if output_count == 0 {
            return Err(SerializationError::Conversion(
                "classification output count must be positive".to_owned(),
            ));
        }
        Ok(Self::new(
            forest_from_dto(dto.forest, ModelKindDto::Classification, Some(output_count))?,
            output_count,
        ))
    }
}

impl TryFrom<&super::RegressionForest> for RegressionModelDtoV1 {
    type Error = SerializationError;

    fn try_from(model: &super::RegressionForest) -> Result<Self, Self::Error> {
        if model.output_count() == 0 {
            return Err(SerializationError::Conversion(
                "regression output count must be positive".to_owned(),
            ));
        }
        Ok(Self {
            forest: forest_to_dto(
                model.forest(),
                ModelKindDto::Regression,
                Some(model.output_count()),
            )?,
            output_count: u32::try_from(model.output_count())
                .map_err(|_| SerializationError::LengthOverflow)?,
        })
    }
}

impl TryFrom<RegressionModelDtoV1> for super::RegressionForest {
    type Error = SerializationError;

    fn try_from(dto: RegressionModelDtoV1) -> Result<Self, Self::Error> {
        let output_count =
            usize::try_from(dto.output_count).map_err(|_| SerializationError::LengthOverflow)?;
        if output_count == 0 {
            return Err(SerializationError::Conversion(
                "regression output count must be positive".to_owned(),
            ));
        }
        Ok(Self::new(
            forest_from_dto(dto.forest, ModelKindDto::Regression, Some(output_count))?,
            output_count,
        ))
    }
}

impl TryFrom<&super::IsolationForest> for IsolationModelDtoV1 {
    type Error = SerializationError;

    fn try_from(model: &super::IsolationForest) -> Result<Self, Self::Error> {
        if model.normalization_size() == 0 {
            return Err(SerializationError::Conversion(
                "isolation normalization size must be positive".to_owned(),
            ));
        }
        Ok(Self {
            forest: forest_to_dto(model.forest(), ModelKindDto::Isolation, None)?,
            normalization_size: usize_to_u64(model.normalization_size())?,
        })
    }
}

impl TryFrom<IsolationModelDtoV1> for super::IsolationForest {
    type Error = SerializationError;

    fn try_from(dto: IsolationModelDtoV1) -> Result<Self, Self::Error> {
        let normalization_size =
            u64_to_usize(dto.normalization_size, "isolation normalization size")?;
        if normalization_size == 0 {
            return Err(SerializationError::Conversion(
                "isolation normalization size must be positive".to_owned(),
            ));
        }
        Ok(Self::new(
            forest_from_dto(dto.forest, ModelKindDto::Isolation, None)?,
            normalization_size,
        ))
    }
}

impl TryFrom<&super::ClassificationForest> for ModelDtoV1 {
    type Error = SerializationError;
    fn try_from(model: &super::ClassificationForest) -> Result<Self, Self::Error> {
        Ok(Self::Classification(ClassificationModelDtoV1::try_from(
            model,
        )?))
    }
}

impl TryFrom<&super::RegressionForest> for ModelDtoV1 {
    type Error = SerializationError;
    fn try_from(model: &super::RegressionForest) -> Result<Self, Self::Error> {
        Ok(Self::Regression(RegressionModelDtoV1::try_from(model)?))
    }
}

impl TryFrom<&super::IsolationForest> for ModelDtoV1 {
    type Error = SerializationError;
    fn try_from(model: &super::IsolationForest) -> Result<Self, Self::Error> {
        Ok(Self::Isolation(IsolationModelDtoV1::try_from(model)?))
    }
}

fn as_core_error(error: SerializationError) -> crate::CoreError {
    crate::CoreError::invalid_data(format!(
        "invalid serialized proximity-forest model: {error}"
    ))
}

pub(crate) fn classification_to_bytes(
    model: &super::ClassificationForest,
) -> crate::CoreResult<Vec<u8>> {
    let dto = ModelDtoV1::try_from(model).map_err(as_core_error)?;
    encode_model_v1(&dto).map_err(as_core_error)
}

pub(crate) fn classification_from_bytes(
    bytes: &[u8],
) -> crate::CoreResult<super::ClassificationForest> {
    match decode_model_v1(bytes).map_err(as_core_error)? {
        ModelDtoV1::Classification(dto) => {
            super::ClassificationForest::try_from(dto).map_err(as_core_error)
        }
        _ => Err(crate::CoreError::invalid_data(
            "serialized model is not a classification forest",
        )),
    }
}

pub(crate) fn regression_to_bytes(model: &super::RegressionForest) -> crate::CoreResult<Vec<u8>> {
    let dto = ModelDtoV1::try_from(model).map_err(as_core_error)?;
    encode_model_v1(&dto).map_err(as_core_error)
}

pub(crate) fn regression_from_bytes(bytes: &[u8]) -> crate::CoreResult<super::RegressionForest> {
    match decode_model_v1(bytes).map_err(as_core_error)? {
        ModelDtoV1::Regression(dto) => {
            super::RegressionForest::try_from(dto).map_err(as_core_error)
        }
        _ => Err(crate::CoreError::invalid_data(
            "serialized model is not a regression forest",
        )),
    }
}

pub(crate) fn isolation_to_bytes(model: &super::IsolationForest) -> crate::CoreResult<Vec<u8>> {
    let dto = ModelDtoV1::try_from(model).map_err(as_core_error)?;
    encode_model_v1(&dto).map_err(as_core_error)
}

pub(crate) fn isolation_from_bytes(bytes: &[u8]) -> crate::CoreResult<super::IsolationForest> {
    match decode_model_v1(bytes).map_err(as_core_error)? {
        ModelDtoV1::Isolation(dto) => super::IsolationForest::try_from(dto).map_err(as_core_error),
        _ => Err(crate::CoreError::invalid_data(
            "serialized model is not an isolation forest",
        )),
    }
}

impl TryFrom<&super::ForestConfig> for ForestConfigDtoV1 {
    type Error = SerializationError;

    #[allow(unreachable_patterns)]
    fn try_from(config: &super::ForestConfig) -> Result<Self, Self::Error> {
        let dimension_selection = match config.dimension_selection {
            super::DimensionSelection::All => DimensionSelectionDtoV1::All,
            super::DimensionSelection::Sqrt => DimensionSelectionDtoV1::SquareRoot,
            super::DimensionSelection::Log2 => DimensionSelectionDtoV1::Log2,
            super::DimensionSelection::FixedCount(value) => DimensionSelectionDtoV1::FixedCount(
                u64::try_from(value).map_err(|_| SerializationError::LengthOverflow)?,
            ),
            super::DimensionSelection::Proportion(value) => {
                DimensionSelectionDtoV1::Fraction(value)
            }
            _ => {
                return Err(SerializationError::Conversion(
                    "unsupported future dimension-selection variant".to_owned(),
                ));
            }
        };
        let branch_tie_policy = match config.branch_tie_policy {
            super::TiePolicy::Random => TiePolicyDtoV1::Random,
            super::TiePolicy::First => TiePolicyDtoV1::First,
        };
        let data_retention = match config.data_retention {
            super::DataRetention::Exemplars => DataRetentionDtoV1::Exemplars,
            super::DataRetention::Copy => DataRetentionDtoV1::Copy,
            super::DataRetention::Reference => DataRetentionDtoV1::Reference,
        };
        Ok(Self {
            tree_count: u64::try_from(config.tree_count.get())
                .map_err(|_| SerializationError::LengthOverflow)?,
            candidate_count: u64::try_from(config.candidate_count.get())
                .map_err(|_| SerializationError::LengthOverflow)?,
            bootstrap: config.bootstrap,
            maximum_depth: config
                .maximum_depth
                .map(u64::try_from)
                .transpose()
                .map_err(|_| SerializationError::LengthOverflow)?,
            minimum_split_size: u64::try_from(config.minimum_split_size.get())
                .map_err(|_| SerializationError::LengthOverflow)?,
            dimension_selection,
            seed: config.seed,
            early_abandoning: config.early_abandoning,
            branch_tie_policy,
            data_retention,
        })
    }
}

impl TryFrom<ForestConfigDtoV1> for super::ForestConfig {
    type Error = SerializationError;

    fn try_from(dto: ForestConfigDtoV1) -> Result<Self, Self::Error> {
        use std::num::NonZeroUsize;

        let to_usize = |value: u64, field: &str| {
            usize::try_from(value).map_err(|_| {
                SerializationError::Conversion(format!("{field} does not fit in usize"))
            })
        };
        let tree_count =
            NonZeroUsize::new(to_usize(dto.tree_count, "tree_count")?).ok_or_else(|| {
                SerializationError::Conversion("tree_count must be positive".to_owned())
            })?;
        let candidate_count = NonZeroUsize::new(to_usize(dto.candidate_count, "candidate_count")?)
            .ok_or_else(|| {
                SerializationError::Conversion("candidate_count must be positive".to_owned())
            })?;
        let minimum_split_size =
            NonZeroUsize::new(to_usize(dto.minimum_split_size, "minimum_split_size")?).ok_or_else(
                || SerializationError::Conversion("minimum_split_size must be positive".to_owned()),
            )?;
        let dimension_selection = match dto.dimension_selection {
            DimensionSelectionDtoV1::All => super::DimensionSelection::All,
            DimensionSelectionDtoV1::SquareRoot => super::DimensionSelection::Sqrt,
            DimensionSelectionDtoV1::Log2 => super::DimensionSelection::Log2,
            DimensionSelectionDtoV1::FixedCount(value) => {
                super::DimensionSelection::FixedCount(to_usize(value, "fixed dimension count")?)
            }
            DimensionSelectionDtoV1::Fraction(value) => {
                super::DimensionSelection::Proportion(value)
            }
        };
        let branch_tie_policy = match dto.branch_tie_policy {
            TiePolicyDtoV1::Random => super::TiePolicy::Random,
            TiePolicyDtoV1::First => super::TiePolicy::First,
        };
        let data_retention = match dto.data_retention {
            DataRetentionDtoV1::Exemplars => super::DataRetention::Exemplars,
            DataRetentionDtoV1::Copy => super::DataRetention::Copy,
            DataRetentionDtoV1::Reference => super::DataRetention::Reference,
        };
        let mut config = super::ForestConfig::new(tree_count, candidate_count, dto.seed);
        config.bootstrap = dto.bootstrap;
        config.maximum_depth = dto
            .maximum_depth
            .map(|value| to_usize(value, "maximum_depth"))
            .transpose()?;
        config.minimum_split_size = minimum_split_size;
        config.dimension_selection = dimension_selection;
        config.early_abandoning = dto.early_abandoning;
        config.branch_tie_policy = branch_tie_policy;
        config.data_retention = data_retention;
        config.external_training_data = None;
        if data_retention != super::DataRetention::Reference {
            config.validate().map_err(|error| {
                SerializationError::Conversion(format!("invalid forest configuration: {error}"))
            })?;
        } else {
            dimension_selection.validate().map_err(|error| {
                SerializationError::Conversion(format!("invalid dimension selection: {error}"))
            })?;
            if config.maximum_depth == Some(0) {
                return Err(SerializationError::Conversion(
                    "maximum_depth must be positive when specified".to_owned(),
                ));
            }
        }
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> ModelDtoV1 {
        ModelDtoV1::Isolation(IsolationModelDtoV1 {
            forest: ForestDtoV1 {
                config: ForestConfigDtoV1 {
                    tree_count: 1,
                    candidate_count: 1,
                    bootstrap: false,
                    maximum_depth: None,
                    minimum_split_size: 2,
                    dimension_selection: DimensionSelectionDtoV1::All,
                    seed: 42,
                    early_abandoning: true,
                    branch_tie_policy: TiePolicyDtoV1::First,
                    data_retention: DataRetentionDtoV1::Exemplars,
                },
                trees: vec![],
                retained_training_data: RetainedTrainingDataDtoV1::None,
                training_metadata: TrainingMetadataDtoV1 {
                    sample_count: 0,
                    bootstrap: vec![],
                },
            },
            normalization_size: 1,
        })
    }
    #[test]
    fn envelope_round_trip() {
        let dto = fixture();
        assert_eq!(
            decode_model_v1(&encode_model_v1(&dto).unwrap()).unwrap(),
            dto
        );
    }
    #[test]
    fn corruption_is_rejected() {
        let mut bytes = encode_model_v1(&fixture()).unwrap();
        *bytes.last_mut().unwrap() ^= 1;
        assert!(matches!(
            decode_model_v1(&bytes),
            Err(SerializationError::ChecksumMismatch { .. })
        ));
    }
    #[test]
    fn future_version_is_rejected() {
        let mut bytes = encode_model_v1(&fixture()).unwrap();
        bytes[8..12].copy_from_slice(&2u32.to_le_bytes());
        assert_eq!(
            decode_model_v1(&bytes),
            Err(SerializationError::UnsupportedVersion(2))
        );
    }

    #[test]
    fn forest_config_round_trip() {
        use std::num::NonZeroUsize;
        let mut config = super::super::ForestConfig::new(
            NonZeroUsize::new(17).unwrap(),
            NonZeroUsize::new(3).unwrap(),
            91,
        );
        config.bootstrap = false;
        config.maximum_depth = Some(12);
        config.dimension_selection = super::super::DimensionSelection::Proportion(0.5);
        config.branch_tie_policy = super::super::TiePolicy::First;
        config.data_retention = super::super::DataRetention::Copy;
        let dto = ForestConfigDtoV1::try_from(&config).unwrap();
        assert_eq!(super::super::ForestConfig::try_from(dto).unwrap(), config);
    }

    #[test]
    fn zero_config_counts_are_rejected() {
        let mut dto = match fixture() {
            ModelDtoV1::Isolation(model) => model.forest.config,
            _ => unreachable!(),
        };
        dto.tree_count = 0;
        assert!(super::super::ForestConfig::try_from(dto).is_err());
    }
    #[test]
    fn training_metadata_round_trip() {
        let metadata = super::super::TrainingMetadata::new(
            3,
            vec![
                super::super::BootstrapSample::from_parts(vec![0, 0, 2], vec![2, 0, 1], vec![1])
                    .unwrap(),
            ],
        );
        let dto = TrainingMetadataDtoV1::try_from(&metadata).unwrap();
        assert_eq!(
            super::super::TrainingMetadata::try_from(dto).unwrap(),
            metadata
        );
    }

    #[test]
    fn inconsistent_bootstrap_metadata_is_rejected() {
        let dto = BootstrapSampleDtoV1 {
            sampled_indices: vec![0, 0, 2],
            counts: vec![1, 1, 1],
            out_of_bag: Vec::new(),
        };
        assert!(super::super::BootstrapSample::try_from(dto).is_err());
    }

    #[test]
    fn external_descriptor_round_trip() {
        let descriptor = super::super::ExternalTrainingData::new(
            std::path::PathBuf::from("training.npy"),
            super::super::ExternalDataFormat::Npy,
            crate::ScalarKind::F64,
            vec![10, 4].into_boxed_slice(),
            128,
            320,
            448,
            Some("sha256:example".into()),
        );
        let dto = ExternalTrainingDataDtoV1::try_from(&descriptor).unwrap();
        assert_eq!(
            super::super::ExternalTrainingData::try_from(dto).unwrap(),
            descriptor
        );
    }

    #[test]
    fn external_payload_outside_file_is_rejected() {
        let dto = ExternalTrainingDataDtoV1 {
            path_utf8: "training.npy".to_owned(),
            format: "npy".to_owned(),
            scalar_kind: "float64".to_owned(),
            shape: vec![10, 4],
            payload_offset: 128,
            payload_length: 321,
            file_size: 448,
            fingerprint: None,
        };
        assert!(super::super::ExternalTrainingData::try_from(dto).is_err());
    }

    #[test]
    fn vector_stores_round_trip() {
        for store in
            [
                super::super::OwnedObservationStore::Vector(super::super::OwnedVectorStore::F32(
                    vec![vec![1.0, 2.0], vec![3.0]],
                )),
                super::super::OwnedObservationStore::Vector(super::super::OwnedVectorStore::F64(
                    vec![vec![1.0], vec![2.0, 3.0]],
                )),
            ]
        {
            let dto = ObservationStoreDtoV1::try_from(&store).unwrap();
            assert_eq!(
                super::super::OwnedObservationStore::try_from(dto).unwrap(),
                store
            );
        }
    }

    #[test]
    fn multivariate_stores_round_trip() {
        let f32_store = super::super::OwnedObservationStore::Multivariate(
            super::super::OwnedMultivariateStore::F32(vec![
                super::super::OwnedSeries::from_parts(vec![1.0f32, 2.0, 3.0, 4.0], 2, 2).unwrap(),
                super::super::OwnedSeries::from_parts(vec![5.0f32, 6.0, 7.0, 8.0, 9.0, 10.0], 2, 3)
                    .unwrap(),
            ]),
        );
        let f64_store = super::super::OwnedObservationStore::Multivariate(
            super::super::OwnedMultivariateStore::F64(vec![
                super::super::OwnedSeries::from_parts(vec![1.0f64, 2.0, 3.0, 4.0], 2, 2).unwrap(),
            ]),
        );
        for store in [f32_store, f64_store] {
            let dto = ObservationStoreDtoV1::try_from(&store).unwrap();
            assert_eq!(
                super::super::OwnedObservationStore::try_from(dto).unwrap(),
                store
            );
        }
    }

    #[test]
    fn malformed_vector_offsets_are_rejected() {
        let dto = ObservationStoreDtoV1::VectorF64 {
            values: vec![1.0, 2.0],
            offsets: vec![0, 2, 2],
        };
        assert!(super::super::OwnedObservationStore::try_from(dto).is_err());
    }

    #[test]
    fn non_rectangular_multivariate_offsets_are_rejected() {
        let dto = ObservationStoreDtoV1::MultivariateF32 {
            values: vec![1.0, 2.0, 3.0, 4.0, 5.0],
            series_offsets: vec![0, 5],
            dimension_offsets: vec![0, 2, 5],
        };
        assert!(super::super::OwnedObservationStore::try_from(dto).is_err());
    }

    #[test]
    fn retained_training_data_round_trip() {
        let retained =
            super::super::RetainedTrainingData::Owned(super::super::OwnedObservationStore::Vector(
                super::super::OwnedVectorStore::F64(vec![vec![1.0, 2.0]]),
            ));
        let dto = RetainedTrainingDataDtoV1::try_from(&retained).unwrap();
        assert_eq!(
            super::super::RetainedTrainingData::try_from(dto).unwrap(),
            retained
        );
    }

    #[test]
    fn leaf_payloads_round_trip() {
        let payloads = vec![
            super::super::LeafPayload::Empty,
            super::super::LeafPayload::Classification(
                super::super::ClassificationLeaf::from_counts(vec![
                    vec![(0, 3), (2, 1)],
                    vec![(1, 4)],
                ])
                .unwrap(),
            ),
            super::super::LeafPayload::Regression(
                super::super::RegressionLeaf::from_means(vec![1.5, -2.0]).unwrap(),
            ),
            super::super::LeafPayload::Isolation {
                occurrence_count: 7,
            },
        ];
        for payload in payloads {
            let dto = LeafPayloadDtoV1::try_from(&payload).unwrap();
            assert_eq!(super::super::LeafPayload::try_from(dto).unwrap(), payload);
        }
    }

    #[test]
    fn classification_leaf_rejects_duplicate_or_unsorted_classes() {
        let dto = LeafPayloadDtoV1::Classification {
            counts: vec![vec![(2, 1), (2, 3)]],
        };
        assert!(super::super::LeafPayload::try_from(dto).is_err());
    }

    #[test]
    fn classification_leaf_rejects_zero_counts() {
        let dto = LeafPayloadDtoV1::Classification {
            counts: vec![vec![(0, 0)]],
        };
        assert!(super::super::LeafPayload::try_from(dto).is_err());
    }

    #[test]
    fn regression_leaf_rejects_non_finite_means() {
        let dto = LeafPayloadDtoV1::Regression {
            means: vec![f64::NAN],
        };
        assert!(super::super::LeafPayload::try_from(dto).is_err());
    }

    #[test]
    fn isolation_leaf_rejects_zero_occurrences() {
        let dto = LeafPayloadDtoV1::Isolation {
            occurrence_count: 0,
        };
        assert!(super::super::LeafPayload::try_from(dto).is_err());
    }

    #[test]
    fn fitted_leaf_kind_and_width_are_validated() {
        let classification = super::super::LeafPayload::Classification(
            super::super::ClassificationLeaf::from_counts(vec![vec![(0, 1)]]).unwrap(),
        );
        assert!(validate_leaf_payload_for_model(
            &classification,
            ModelKindDto::Classification,
            Some(1),
        )
        .is_ok());
        assert!(validate_leaf_payload_for_model(
            &classification,
            ModelKindDto::Classification,
            Some(2),
        )
        .is_err());
        assert!(
            validate_leaf_payload_for_model(&classification, ModelKindDto::Regression, Some(1),)
                .is_err()
        );
    }
}
