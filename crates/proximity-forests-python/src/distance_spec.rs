//! Python-facing translation of fixed and candidate-sampled distance choices.

use std::collections::HashMap;

use proximity_forests_core::{
    DistanceChoice, DistanceConfig, DtwArowParameters, DtwParameters, ErpParameters, FiniteF64,
    LcssParameters, MsmParameters, NoOverlapPolicy, SampledDistanceFamily, ShapeDtwParameters,
    TweParameters, WeightedDtwParameters,
};
use pyo3::exceptions::PyValueError;
use pyo3::{PyErr, PyResult};

#[allow(dead_code)] // Multivariate is activated by the upcoming 3-D/list-backed binding.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CatalogLayout {
    Vector,
    Multivariate,
}

pub fn parse_catalog(
    tokens: Vec<String>,
    layout: CatalogLayout,
) -> PyResult<Box<[DistanceChoice]>> {
    if tokens.is_empty() {
        return Err(PyValueError::new_err(
            "distances must contain at least one distance specification",
        ));
    }
    tokens
        .into_iter()
        .map(|token| parse_choice(&token, layout))
        .collect::<PyResult<Vec<_>>>()
        .map(Vec::into_boxed_slice)
}

pub fn parse_vector_catalog(tokens: Vec<String>) -> PyResult<Box<[DistanceChoice]>> {
    parse_catalog(tokens, CatalogLayout::Vector)
}

/// Parses a distance catalog for multivariate series input.
pub fn parse_multivariate_catalog(tokens: Vec<String>) -> PyResult<Box<[DistanceChoice]>> {
    parse_catalog(tokens, CatalogLayout::Multivariate)
}

fn parse_choice(token: &str, layout: CatalogLayout) -> PyResult<DistanceChoice> {
    let (raw_name, query) = token.split_once('?').unwrap_or((token, ""));
    let name = normalize(raw_name);
    validate_layout(&name, layout)?;
    let mut parameters = parse_parameters(query)?;
    let explicitly_fixed = parameters.remove("fixed").is_some();

    if !explicitly_fixed && parameters.is_empty() {
        if let Some(family) = sampled_family(&name) {
            return Ok(DistanceChoice::Sampled(family));
        }
    }

    if name.contains("_cv") {
        return Err(PyValueError::new_err(
            "*_cv names are sampled families; use dtw/ddtw with an explicit window for a fixed configuration",
        ));
    }
    validate_parameter_names(&name, &parameters)?;
    parse_fixed(&name, &parameters).map(DistanceChoice::Fixed)
}

fn sampled_family(name: &str) -> Option<SampledDistanceFamily> {
    use SampledDistanceFamily as F;
    Some(match name {
        "dtw" => F::Dtw,
        "dtw_cv" => F::DtwCv,
        "ddtw" => F::Ddtw,
        "ddtw_cv" => F::DdtwCv,
        "wdtw" => F::Wdtw,
        "wddtw" => F::Wddtw,
        "erp" => F::Erp,
        "lcss" => F::Lcss,
        "msm" => F::Msm,
        "twe" => F::Twe,
        "dtw_i" => F::DtwIndependent,
        "dtw_cv_i" => F::DtwCvIndependent,
        "ddtw_i" => F::DdtwIndependent,
        "ddtw_cv_i" => F::DdtwCvIndependent,
        "wdtw_i" => F::WdtwIndependent,
        "wddtw_i" => F::WddtwIndependent,
        "erp_i" => F::ErpIndependent,
        "lcss_i" => F::LcssIndependent,
        "msm_i" => F::MsmIndependent,
        "twe_i" => F::TweIndependent,
        "dtw_d" => F::DtwDependent,
        "dtw_cv_d" => F::DtwCvDependent,
        "ddtw_d" => F::DdtwDependent,
        "ddtw_cv_d" => F::DdtwCvDependent,
        "wdtw_d" => F::WdtwDependent,
        "wddtw_d" => F::WddtwDependent,
        _ => return None,
    })
}

fn parse_fixed(name: &str, p: &HashMap<String, String>) -> PyResult<DistanceConfig> {
    let dtw = || -> PyResult<DtwParameters> { Ok(DtwParameters { window: window(p)? }) };
    let weighted = || -> PyResult<WeightedDtwParameters> {
        Ok(WeightedDtwParameters {
            window: window(p)?,
            stiffness: finite_parameter(p, "stiffness", 0.05)?,
        })
    };
    let erp = || -> PyResult<ErpParameters> {
        Ok(ErpParameters {
            window: window(p)?,
            gap: finite_parameter(p, "gap", 0.0)?,
        })
    };
    let lcss = || -> PyResult<LcssParameters> {
        Ok(LcssParameters {
            window: window(p)?,
            epsilon: finite_parameter(p, "epsilon", 0.01)?,
        })
    };
    let msm = || -> PyResult<MsmParameters> {
        Ok(MsmParameters {
            cost: finite_parameter(p, "cost", 1.0)?,
        })
    };
    let twe = || -> PyResult<TweParameters> {
        Ok(TweParameters {
            stiffness: finite_parameter(p, "stiffness", 0.001)?,
            penalty: finite_parameter(p, "penalty", 1.0)?,
        })
    };
    let shape =
        || -> PyResult<ShapeDtwParameters> { Ok(ShapeDtwParameters { window: window(p)? }) };

    Ok(match name {
        "squared_euclidean" | "sqeuclidean" | "euclidean" | "ed" => {
            DistanceConfig::SquaredEuclidean
        }
        "manhattan" | "l1" => DistanceConfig::Manhattan,
        "cosine" => DistanceConfig::Cosine,
        "cid" => DistanceConfig::Cid,
        "sbd" => DistanceConfig::Sbd,
        "dtw" => DistanceConfig::Dtw(dtw()?),
        "ddtw" => DistanceConfig::Ddtw(dtw()?),
        "wdtw" => DistanceConfig::Wdtw(weighted()?),
        "wddtw" => DistanceConfig::Wddtw(weighted()?),
        "erp" => DistanceConfig::Erp(erp()?),
        "lcss" => DistanceConfig::Lcss(lcss()?),
        "msm" => DistanceConfig::Msm(msm()?),
        "twe" => DistanceConfig::Twe(twe()?),
        "shape_hog_1d_dtw" | "shapehog1ddtw" => DistanceConfig::ShapeHog1dDtw(shape()?),
        "nan_squared_euclidean" | "nan_euclidean" => DistanceConfig::NaNSquaredEuclidean {
            no_overlap: no_overlap_policy(p)?,
        },
        "dtw_arow" => DistanceConfig::DtwArow(DtwArowParameters { window: window(p)? }),
        "squared_euclidean_i" | "euclidean_i" | "ed_i" => {
            DistanceConfig::SquaredEuclideanIndependent
        }
        "manhattan_i" => DistanceConfig::ManhattanIndependent,
        "cid_i" => DistanceConfig::CidIndependent,
        "sbd_i" => DistanceConfig::SbdIndependent,
        "dtw_i" => DistanceConfig::DtwIndependent(dtw()?),
        "ddtw_i" => DistanceConfig::DdtwIndependent(dtw()?),
        "wdtw_i" => DistanceConfig::WdtwIndependent(weighted()?),
        "wddtw_i" => DistanceConfig::WddtwIndependent(weighted()?),
        "erp_i" => DistanceConfig::ErpIndependent(erp()?),
        "lcss_i" => DistanceConfig::LcssIndependent(lcss()?),
        "msm_i" => DistanceConfig::MsmIndependent(msm()?),
        "twe_i" => DistanceConfig::TweIndependent(twe()?),
        "shape_hog_1d_dtw_i" => DistanceConfig::ShapeHog1dDtwIndependent(shape()?),
        "nan_squared_euclidean_i" | "nan_euclidean_i" => {
            DistanceConfig::NaNSquaredEuclideanIndependent {
                no_overlap: no_overlap_policy(p)?,
            }
        }
        "dtw_arow_i" => {
            DistanceConfig::DtwArowIndependent(DtwArowParameters { window: window(p)? })
        }
        "dtw_d" => DistanceConfig::DtwDependent(dtw()?),
        "ddtw_d" => DistanceConfig::DdtwDependent(dtw()?),
        "wdtw_d" => DistanceConfig::WdtwDependent(weighted()?),
        "wddtw_d" => DistanceConfig::WddtwDependent(weighted()?),
        "shape_hog_dtw" | "shape_hog_dtw_d" => DistanceConfig::ShapeHogDtw(shape()?),
        "dtw_arow_d" => DistanceConfig::DtwArowDependent(DtwArowParameters { window: window(p)? }),
        _ => return Err(unsupported_distance(name)),
    })
}

fn validate_layout(name: &str, layout: CatalogLayout) -> PyResult<()> {
    let multivariate = name.ends_with("_i") || name.ends_with("_d");
    match (layout, multivariate) {
        (CatalogLayout::Vector, true) => Err(PyValueError::new_err(format!(
            "distance {name:?} requires multivariate input"
        ))),
        (CatalogLayout::Multivariate, false) => Err(PyValueError::new_err(format!(
            "distance {name:?} requires an explicit _i or _d form"
        ))),
        _ => Ok(()),
    }
}

fn parse_parameters(query: &str) -> PyResult<HashMap<String, String>> {
    let mut result = HashMap::new();
    if query.is_empty() {
        return Ok(result);
    }
    for pair in query.split('&') {
        let (key, value) = pair
            .split_once('=')
            .ok_or_else(|| PyValueError::new_err(format!("invalid distance parameter {pair:?}")))?;
        if result.insert(key.to_owned(), value.to_owned()).is_some() {
            return Err(PyValueError::new_err(format!(
                "duplicate distance parameter {key:?}"
            )));
        }
    }
    Ok(result)
}

fn validate_parameter_names(name: &str, p: &HashMap<String, String>) -> PyResult<()> {
    let allowed: &[&str] = match name {
        "dtw" | "ddtw" | "dtw_i" | "ddtw_i" | "dtw_d" | "ddtw_d" => &["window"],
        "wdtw" | "wddtw" | "wdtw_i" | "wddtw_i" | "wdtw_d" | "wddtw_d" => &["window", "stiffness"],
        "erp" | "erp_i" => &["window", "gap"],
        "lcss" | "lcss_i" => &["window", "epsilon"],
        "msm" | "msm_i" => &["cost"],
        "twe" | "twe_i" => &["stiffness", "penalty"],
        "shape_hog_1d_dtw" | "shape_hog_1d_dtw_i" | "shape_hog_dtw" | "shape_hog_dtw_d"
        | "dtw_arow" | "dtw_arow_i" | "dtw_arow_d" => &["window"],
        "nan_squared_euclidean"
        | "nan_euclidean"
        | "nan_squared_euclidean_i"
        | "nan_euclidean_i" => &["no_overlap"],
        _ => &[],
    };
    if let Some(key) = p.keys().find(|key| !allowed.contains(&key.as_str())) {
        return Err(PyValueError::new_err(format!(
            "parameter {key:?} is not valid for distance {name:?}"
        )));
    }
    Ok(())
}

fn window(p: &HashMap<String, String>) -> PyResult<Option<usize>> {
    p.get("window")
        .map(|v| {
            v.parse::<usize>()
                .map_err(|_| PyValueError::new_err("distance window must be a nonnegative integer"))
        })
        .transpose()
}

fn finite_parameter(p: &HashMap<String, String>, name: &str, default: f64) -> PyResult<FiniteF64> {
    let value = p.get(name).map_or(Ok(default), |v| {
        v.parse::<f64>().map_err(|_| {
            PyValueError::new_err(format!("distance parameter {name:?} must be numeric"))
        })
    })?;
    FiniteF64::new(value).map_err(|error| PyValueError::new_err(error.to_string()))
}

fn normalize(name: &str) -> String {
    name.trim().to_ascii_lowercase().replace(['-', ' '], "_")
}
fn unsupported_distance(token: &str) -> PyErr {
    PyValueError::new_err(format!("unsupported distance {token:?}; see DISTANCES.md"))
}

fn no_overlap_policy(p: &HashMap<String, String>) -> PyResult<NoOverlapPolicy> {
    match p
        .get("no_overlap")
        .map(String::as_str)
        .unwrap_or("infinite")
    {
        "infinite" => Ok(NoOverlapPolicy::Infinite),
        "zero" => Ok(NoOverlapPolicy::Zero),
        value => Err(PyValueError::new_err(format!(
            "no_overlap must be 'infinite' or 'zero'; received {value:?}"
        ))),
    }
}

/// Rejects catalogs that could select a non-missing-compatible distance when
/// the current input contains NaN. Complete input accepts any valid mixture.
pub fn validate_missing_compatibility(
    catalog: &[DistanceChoice],
    contains_nan: bool,
) -> PyResult<()> {
    if !contains_nan {
        return Ok(());
    }
    for choice in catalog {
        let compatible = match choice {
            DistanceChoice::Fixed(config) => config.capabilities().missing_values,
            DistanceChoice::Sampled(_) => false,
        };
        if !compatible {
            return Err(PyValueError::new_err(
                "X contains NaN, so every configured distance must support missing values",
            ));
        }
    }
    Ok(())
}
