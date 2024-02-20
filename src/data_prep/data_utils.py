import re
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path

import exiftool


@dataclass
class MetaData:
    image_path: str
    sample_type: str = field(init=False)  # BENI: benitier, SED: sediment, CSED: controle sediment, CBENI: ...
    island: str = field(init=False)  # AR: arutua, MAK: makemo, ...
    station: str = field(init=False)  # Si
    replica: str = field(init=False)  # Ri
    distil: str = ""  # Di
    sample_id: str = field(init=False)  # Fi
    image_id: str = field(init=False)  # iiii
    filter: str = ""  # DAPI, CY2, NAT, TRI
    extra: str = ""  # bis ou cut
    # number: int = 0
    date: str = field(init=False)
    exposure_time: float = 0.0

    def __post_init__(self):
        self.parse_metadata_from_name()
        self.parse_metadata_from_exif()

    def parse_metadata_from_name(self):
        base_name = Path(self.image_path).stem
        parts = base_name.split('_')
        self.sample_type = parts[0]
        self.island = parts[1]
        self.station = parts[2]
        sid_match = re.findall(r"_F\d_", base_name)
        self.sample_id = sid_match[0][1:-1] if sid_match else "UNK"
        for exp in [r"(_R\d_)", r"(_\dV_)", r"(_\dB_)", r"(_\dH_)"]:
            rep_match = re.findall(exp, base_name)
            if rep_match:
                self.replica = rep_match[0][1:-1]
                break
            else:
                self.replica = "UNK"
        d_match = re.findall(r"_D\d_", base_name)
        self.distil = d_match[0][1:-1] if d_match else "UNK"
        im_id_match = re.findall(r"\d{4}", base_name)
        self.image_id = im_id_match[0] if im_id_match else ""
        # number_match = re.findall(r"\(\d{1,3}\)", str(base_name))
        # self.number = int(number_match[0][1:-1]) if number_match else 0
        suffix_match = re.findall(r"\d{4}_\w+", base_name)
        if suffix_match:
            if "bis" in suffix_match[0].lower():
                self.extra = "BIS"
            elif "cut" in suffix_match[0].lower():
                self.extra = "CUT"
            else:
                self.extra = ""
            if "dapi" in suffix_match[0].lower():
                self.filter = "DAPI"
            elif "nat" in suffix_match[0].lower():
                self.filter = "NAT"
            elif "tri" in suffix_match[0].lower():
                self.filter = "TRI"
            elif "cy" in suffix_match[0].lower():
                self.filter = "CY2"
            else:
                self.filter = ""

    def parse_metadata_from_exif(self):
        with exiftool.ExifToolHelper() as et:
            extra_metadata = et.get_metadata(self.image_path)[0]
            date = extra_metadata.get("EXIF:CreateDate", "NoDate")
            self.date = date.split(' ')[0].replace(":", '-')  # self.date = date.replace(":", '-').replace(' ', '-')
            self.exposure_time = float(extra_metadata.get("EXIF:ExposureTime", "0.0"))


def get_obs_id(sample):
    # get prefix that constitute unique identifier of the observation
    return "_".join([sample.date,
                     sample.sample_type, sample.island, sample.station, sample.replica, sample.sample_id,
                     sample.distil,
                     sample.image_id])


class DataPrior(IntEnum):
    RENAMED = 1  # ex : lot 1
    PARTIAL_RENAMED = 2  # ex lot2 (only obs id renamed ie: CSED_TAK_S1_UNK_UNK_0001 (1).jpg)
    CONSECUTIVE = 3  # ex lot3 (not renamed but obs are organized in 4 consecutive images
    # ie: BENI_MAK_S1_5V_F2_UNK_0000.jpg)
