import pandas as pd
import os

from django.http import JsonResponse, FileResponse, Http404
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.utils.text import slugify
from ..models import Job

@csrf_exempt
def detect_csv_format(request):
    if request.method != "POST" or "file" not in request.FILES:
        return JsonResponse({"error": "CSV file is missing."}, status=400)
    file = request.FILES["file"]
    try:
        df = pd.read_csv(file)
        df = df.dropna(how="all")  # Remove empty rows
    except Exception as e:
        return JsonResponse(
            {"status": "invalid", "errors": [f"Error reading CSV: {str(e)}"]},
            status=400,
        )

    errors = []
    required_cols = {"Protein Sequence"}
    has_substrate = "Substrate" in df.columns
    has_substrates_products = "Substrates" in df.columns and "Products" in df.columns

    if not required_cols.issubset(df.columns):
        errors.append("Missing required column: 'Protein Sequence'")

    if not has_substrate and not has_substrates_products:
        errors.append(
            "Missing substrate information: expected either 'Substrate' or both 'Substrates' and 'Products'"
        )

    if errors:
        return JsonResponse({"status": "invalid", "errors": errors})

    if has_substrates_products and has_substrate:
        return JsonResponse(
            {
                "status": "invalid",
                "errors": [
                    "Cannot have both 'Substrate' and 'Substrates'/'Products' columns."
                ],
            },
            status=400,
        )

    valid_response = {"status": "valid", "num_rows": len(df)}
    if has_substrates_products:
        valid_response["csv_type"] = "multi"
    else:
        if not has_substrate:
            return JsonResponse(
                {
                    "status": "invalid",
                    "errors": [
                        "Could not determine CSV format. Read instructions and check the example CSV files."
                    ],
                },
                status=400,
            )
        valid_response["csv_type"] = "single"
    return JsonResponse(valid_response, status=200)


def download_job_output(request, public_id):
    try:
        job = Job.objects.get(public_id=public_id)
    except Job.DoesNotExist:
        raise Http404("Job not found.")

    media_url = settings.MEDIA_ROOT + f"/{job.output_file.name}"

    if not os.path.exists(media_url):
        raise Http404("No output file for this job.")

    slugified_id = slugify(str(public_id))
    default_name = f"job-{slugified_id}-output.csv"

    response = FileResponse(
        open(media_url, "rb"), as_attachment=True, filename=default_name
    )
    response["Content-Type"] = "text/csv"
    return response


def download_job_input(request, public_id):
    try:
        job = Job.objects.get(public_id=public_id)
    except Job.DoesNotExist:
        raise Http404("Job not found.")

    media_url = f"{settings.MEDIA_ROOT}/jobs/{job.public_id}/input.csv"
    if not os.path.exists(media_url):
        raise Http404("No input file for this job.")

    slugified_id = slugify(str(public_id))
    default_name = f"job-{slugified_id}-input.csv"

    response = FileResponse(
        open(media_url, "rb"), as_attachment=True, filename=default_name
    )
    response["Content-Type"] = "text/csv"
    return response
