class ForecastCoreError(Exception):
    """Base exception for predictable Forecast Core failures."""


class DataValidationError(ForecastCoreError):
    """Raised when canonical input cannot be made safe for modelling."""


class InsufficientDataError(ForecastCoreError):
    """Raised when there is not enough chronological history."""


class ArtifactError(ForecastCoreError):
    """Raised when model artifacts are missing or incompatible."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "ARTIFACT_ERROR",
        details: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details = dict(details or {})

    def to_dict(self) -> dict[str, object]:
        return {"code": self.code, "message": str(self), "details": self.details}


class FeatureSchemaError(ForecastCoreError):
    """Raised when runtime features do not match the saved schema."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "FEATURE_SCHEMA_MISMATCH",
        details: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details = dict(details or {})

    def to_dict(self) -> dict[str, object]:
        return {"code": self.code, "message": str(self), "details": self.details}


class BOMError(ForecastCoreError):
    """Base exception for deterministic Recipe/BOM failures."""

    default_code = "BOM_ERROR"

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        details: dict[str, object] | None = None,
        recoverable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code or self.default_code
        self.details = dict(details or {})
        self.recoverable = recoverable

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation for API boundaries."""

        return {
            "code": self.code,
            "message": str(self),
            "details": self.details,
            "recoverable": self.recoverable,
        }


class RecipeValidationError(BOMError):
    """Raised when canonical recipe input is invalid."""

    default_code = "INVALID_RECIPE"


class RecipeVersionError(BOMError):
    """Raised when active recipe version selection is ambiguous."""

    default_code = "INVALID_RECIPE_VERSION"


class UnitConversionError(BOMError):
    """Raised when no safe unit conversion path exists."""

    default_code = "UNIT_CONVERSION_NOT_SUPPORTED"


class InvalidUnitConversionError(BOMError):
    """Raised when supplied unit-conversion metadata is invalid."""

    default_code = "INVALID_UNIT_CONVERSION"


class ProductUnitConsistencyError(BOMError):
    """Raised when one store-product series declares multiple product units."""

    default_code = "INCONSISTENT_PRODUCT_UNIT"


class ScenarioError(ForecastCoreError):
    """Base exception for probabilistic scenario failures."""

    default_code = "SCENARIO_ERROR"

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        details: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code or self.default_code
        self.details = dict(details or {})

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": str(self),
            "details": self.details,
        }


class ScenarioValidationError(ScenarioError):
    """Raised when a scenario contract or table is structurally invalid."""

    default_code = "SCENARIO_VALIDATION_ERROR"


class ScenarioDataInsufficiencyError(ScenarioError):
    """Raised when genuine out-of-sample history cannot support scenarios."""

    default_code = "SCENARIO_HISTORY_INSUFFICIENT"


class InventoryError(ForecastCoreError):
    """Base structured exception for inventory consequence failures."""

    default_code = "INVENTORY_ERROR"

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        details: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code or self.default_code
        self.details = dict(details or {})

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": str(self),
            "details": self.details,
        }


class InventoryValidationError(InventoryError):
    default_code = "INVENTORY_VALIDATION_ERROR"


class InventoryAccountingError(InventoryError):
    default_code = "INVENTORY_ACCOUNTING_VIOLATION"


class UnknownExpiryError(InventoryError):
    default_code = "UNKNOWN_EXPIRY"


class OptimizationError(ForecastCoreError):
    """Base structured exception for procurement optimization failures."""

    default_code = "OPTIMIZATION_ERROR"

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        details: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code or self.default_code
        self.details = dict(details or {})

    def to_dict(self) -> dict[str, object]:
        return {"code": self.code, "message": str(self), "details": self.details}


class InfeasibleProcurementError(OptimizationError):
    default_code = "INFEASIBLE_PROCUREMENT"


class OptimizationNotAvailableError(OptimizationError):
    default_code = "OPTIMIZATION_NOT_AVAILABLE"
