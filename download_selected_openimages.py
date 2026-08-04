from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError


IMAGE_IDS = [
    "d1f593b12c5e9d92",
    "583a3460c32053a1",
    "4328dda1d3b560d5",
    "886c494f2e608b89",
]

POSSIBLE_SPLITS = [
    "train",
    "validation",
    "test",
    "challenge2018",
]

BUCKET_NAME = "open-images-dataset"

OUTPUT_DIRECTORY = Path(
    "data/images/openimages"
)


def download_image(
    s3_client,
    image_id: str,
) -> bool:
    """
    Try each Open Images split until the image is found.

    Returns True when the image is downloaded successfully.
    """

    output_path = (
        OUTPUT_DIRECTORY
        / f"{image_id}.jpg"
    )

    if output_path.exists():
        print(
            f"Already exists: {output_path}"
        )
        return True

    for split in POSSIBLE_SPLITS:
        object_key = (
            f"{split}/{image_id}.jpg"
        )

        print(
            f"Trying {object_key}..."
        )

        try:
            s3_client.download_file(
                BUCKET_NAME,
                object_key,
                str(output_path),
            )

        except ClientError as error:
            error_code = str(
                error.response
                .get("Error", {})
                .get("Code", "")
            )

            if output_path.exists():
                output_path.unlink()

            if error_code in {
                "404",
                "NoSuchKey",
                "NotFound",
            }:
                continue

            raise RuntimeError(
                f"Unexpected S3 error while "
                f"downloading {object_key}: "
                f"{error_code}"
            ) from error

        print(
            f"Downloaded: {object_key}"
        )

        print(
            f"Saved to: {output_path}"
        )

        return True

    print(
        f"Not found in any split: "
        f"{image_id}"
    )

    return False


def main() -> None:
    """Download the selected Open Images files."""

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    s3_client = boto3.client(
        "s3",
        config=Config(
            signature_version=UNSIGNED
        ),
    )

    successful_downloads = 0

    for image_id in IMAGE_IDS:
        print("\n" + "-" * 70)

        downloaded = download_image(
            s3_client=s3_client,
            image_id=image_id,
        )

        if downloaded:
            successful_downloads += 1

    print("\n" + "=" * 70)

    print(
        f"Downloaded successfully: "
        f"{successful_downloads}/"
        f"{len(IMAGE_IDS)}"
    )

    print(
        f"Output directory: "
        f"{OUTPUT_DIRECTORY}"
    )


if __name__ == "__main__":
    main()