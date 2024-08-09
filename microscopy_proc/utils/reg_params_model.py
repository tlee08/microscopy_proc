from pydantic import BaseModel, ConfigDict, model_validator


class RegParamsModel(BaseModel):
    """
    Pydantic model for registration parameters
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    ref_orient_ls: tuple = (1, 2, 3)
    ref_z_trim: tuple = (None, None, None)
    ref_y_trim: tuple = (None, None, None)
    ref_x_trim: tuple = (None, None, None)
    z_rough: int = 3
    y_rough: int = 6
    x_rough: int = 6
    z_fine: float = 1.0
    y_fine: float = 0.6
    x_fine: float = 0.6
    z_trim: tuple = (None, None, None)
    y_trim: tuple = (None, None, None)
    x_trim: tuple = (None, None, None)

    @model_validator(mode="after")
    def validate_trims(self):
        # Orient validation
        assert len(self.ref_orient_ls) == 3
        assert all(isinstance(i, int) for i in self.ref_orient_ls)
        ref_orient_ls_abs = [abs(i) for i in self.ref_orient_ls]
        assert max(ref_orient_ls_abs) == 3
        assert min(ref_orient_ls_abs) == 1
        assert sum(ref_orient_ls_abs) == 6
        # Trim validation
        for i in (
            self.ref_z_trim,
            self.ref_y_trim,
            self.ref_x_trim,
            self.z_trim,
            self.y_trim,
            self.x_trim,
        ):
            assert len(i) == 3
            assert all(isinstance(j, int) or j is None for j in i)
        return self
