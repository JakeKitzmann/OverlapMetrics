import itk
import pandas as pd
import numpy as np
import sys
import os
from glob import glob 
from pathlib import Path

class OverlapResults:
    def __init__(self, dsc, fpr, tpr):
        self.dsc = dsc
        self.fpr = fpr
        self.tpr = tpr

def process(image_a, image_b):

    Dimension = 3
    LabelPixelType = itk.UC
    LabelImageType = itk.Image[LabelPixelType, Dimension]

    source_reader = itk.ImageFileReader[LabelImageType].New()
    source_reader.SetFileName(image_a)
    source_reader.Update()

    target_reader = itk.ImageFileReader[LabelImageType].New()
    target_reader.SetFileName(image_b)
    target_reader.Update()

    overlap_filter = itk.LabelOverlapMeasuresImageFilter[LabelImageType].New()
    overlap_filter.SetSourceImage(source_reader.GetOutput())
    overlap_filter.SetTargetImage(target_reader.GetOutput())
    overlap_filter.Update()

    return OverlapResults(
        overlap_filter.GetDiceCoefficient(),
        overlap_filter.GetFalseNegativeError(),
        overlap_filter.GetFalsePositiveError()
    )


def main():
    dir_a = Path(sys.argv[1])
    dir_b = Path(sys.argv[2])

    scans_a = sorted(dir_a.rglob("*.nii.gz"))
    scans_b = sorted(dir_b.rglob("*.nii.gz"))


    if len(scans_a) != len(scans_b):
        raise ValueError(
            f"Different scan counts: {len(scans_a)} in A, "
            f"{len(scans_b)} in B"
        )

    rows = []

    for image_a, image_b in zip(scans_a, scans_b):
        print(f'assessing {image_a} vs {image_b}')
        results = process(str(image_a), str(image_b))

        rows.append({
            "image_a": str(image_a),
            "image_b": str(image_b),
            "dsc": results.dsc,
            "fpr": results.fpr,
            "tpr": results.tpr,
        })

    df = pd.DataFrame(rows)

    print(df)
    df.to_csv("overlap_results.csv", index=False)

if __name__ == "__main__":
    main()