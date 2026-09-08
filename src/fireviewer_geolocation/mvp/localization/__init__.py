"""Part.3 coarse localization primitives that do not depend on the map builder."""

from fireviewer_geolocation.mvp.localization.azure_maps import (
    AzureIdentityMapsTransport,
    AzureMapsConfig,
    AzureMapsEnrichmentRun,
    AzureMapsError,
    AzureMapsGeoEnrichmentProvider,
    AzureMapsLocationQuery,
)
from fireviewer_geolocation.mvp.localization.durable_terrain import (
    AzureBackendTerrainResolver,
    DurableTerrainError,
    DurableTerrainTransport,
    TerrainDownloadReceipt,
    UrllibDurableTerrainTransport,
)
from fireviewer_geolocation.mvp.localization.event_localizer import (
    EventLocalizationConfig,
    LocalEvidenceImageLoader,
    MegaLocFaissEventLocalizer,
    abstain_for_missing_reference_coverage,
)
from fireviewer_geolocation.mvp.localization.evidence_fusion import (
    DeterministicEvidenceFusion,
    FusionConfig,
    FusionWeights,
    haversine_m,
)
from fireviewer_geolocation.mvp.localization.geographic_endpoint import (
    DurableGeographicHypothesisService,
    create_geographic_hypothesis_server,
)
from fireviewer_geolocation.mvp.localization.geographic_hypotheses import (
    GeographicHypothesisConfig,
    GeographicHypothesisEngine,
    TerrainElevationProvider,
    TerrainSurfaceElevationProvider,
    TerrainVisibility,
)
from fireviewer_geolocation.mvp.localization.local_megaloc_bundle import (
    LocalMegaLocBundleManifest,
    LocalMegaLocModelLoader,
    inspect_local_megaloc_bundle,
)
from fireviewer_geolocation.mvp.localization.panoramax_cache import (
    CachedPanoramaxImageLoader,
    PanoramaxCacheManifest,
    materialize_panoramax_cache,
)
from fireviewer_geolocation.mvp.localization.regional_index import (
    CallablePanoramaxImageLoader,
    PanoramaxRegionalIndexBuilder,
    RegionalIndexConfig,
)
from fireviewer_geolocation.mvp.localization.retrieval import MegaLocFaissRetriever, RetrievalConfig

__all__ = [
    "AzureBackendTerrainResolver",
    "AzureIdentityMapsTransport",
    "AzureMapsConfig",
    "AzureMapsEnrichmentRun",
    "AzureMapsError",
    "AzureMapsGeoEnrichmentProvider",
    "AzureMapsLocationQuery",
    "CachedPanoramaxImageLoader",
    "CallablePanoramaxImageLoader",
    "DeterministicEvidenceFusion",
    "DurableGeographicHypothesisService",
    "DurableTerrainError",
    "DurableTerrainTransport",
    "EventLocalizationConfig",
    "FusionConfig",
    "FusionWeights",
    "GeographicHypothesisConfig",
    "GeographicHypothesisEngine",
    "LocalEvidenceImageLoader",
    "LocalMegaLocBundleManifest",
    "LocalMegaLocModelLoader",
    "MegaLocFaissEventLocalizer",
    "MegaLocFaissRetriever",
    "PanoramaxCacheManifest",
    "PanoramaxRegionalIndexBuilder",
    "RegionalIndexConfig",
    "RetrievalConfig",
    "TerrainDownloadReceipt",
    "TerrainElevationProvider",
    "TerrainSurfaceElevationProvider",
    "TerrainVisibility",
    "UrllibDurableTerrainTransport",
    "abstain_for_missing_reference_coverage",
    "create_geographic_hypothesis_server",
    "haversine_m",
    "inspect_local_megaloc_bundle",
    "materialize_panoramax_cache",
]
