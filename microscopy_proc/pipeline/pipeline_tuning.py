from microscopy_proc.pipeline.pipeline import Pipeline
from microscopy_proc.utils.logging_utils import init_logger
from microscopy_proc.utils.proj_org_utils import ProjFpModel

MSG_TO_TUNING_PFM = "Converting pfm to tuning filepaths (copy)"


class PipelineTuning(Pipeline):
    logger = init_logger()

    ###################################################################################################
    # RUN CELL COUNTING PIPELINE WITH TUNING FILEPATHS
    ###################################################################################################

    @classmethod
    def img_overlap(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        cls.logger.debug(MSG_TO_TUNING_PFM)
        super().img_overlap(pfm.copy().convert_to_tuning(), overwrite=overwrite)

    @classmethod
    def cellc1(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        cls.logger.debug(MSG_TO_TUNING_PFM)
        super().cellc1(pfm.copy().convert_to_tuning(), overwrite=overwrite)

    @classmethod
    def cellc2(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        cls.logger.debug(MSG_TO_TUNING_PFM)
        super().cellc2(pfm.copy().convert_to_tuning(), overwrite=overwrite)

    @classmethod
    def cellc3(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        cls.logger.debug(MSG_TO_TUNING_PFM)
        super().cellc3(pfm.copy().convert_to_tuning(), overwrite=overwrite)

    @classmethod
    def cellc4(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        cls.logger.debug(MSG_TO_TUNING_PFM)
        super().cellc4(pfm.copy().convert_to_tuning(), overwrite=overwrite)

    @classmethod
    def cellc5(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        cls.logger.debug(MSG_TO_TUNING_PFM)
        super().cellc5(pfm.copy().convert_to_tuning(), overwrite=overwrite)

    @classmethod
    def cellc6(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        cls.logger.debug(MSG_TO_TUNING_PFM)
        super().cellc6(pfm.copy().convert_to_tuning(), overwrite=overwrite)

    @classmethod
    def cellc7(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        cls.logger.debug(MSG_TO_TUNING_PFM)
        super().cellc7(pfm.copy().convert_to_tuning(), overwrite=overwrite)

    @classmethod
    def cellc8(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        cls.logger.debug(MSG_TO_TUNING_PFM)
        super().cellc8(pfm.copy().convert_to_tuning(), overwrite=overwrite)

    @classmethod
    def cellc9(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        cls.logger.debug(MSG_TO_TUNING_PFM)
        super().cellc9(pfm.copy().convert_to_tuning(), overwrite=overwrite)

    @classmethod
    def cellc10(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        cls.logger.debug(MSG_TO_TUNING_PFM)
        super().cellc10(pfm.copy().convert_to_tuning(), overwrite=overwrite)

    @classmethod
    def cellc11(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        cls.logger.debug(MSG_TO_TUNING_PFM)
        super().cellc11(pfm.copy().convert_to_tuning(), overwrite=overwrite)

    @classmethod
    def cellc_coords_only(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        cls.logger.debug(MSG_TO_TUNING_PFM)
        super().cellc_coords_only(pfm.copy().convert_to_tuning(), overwrite=overwrite)
