"""
zenodo_upload.py -- create a Zenodo preprint record for the paper via the Zenodo REST API.

Usage:
    export ZENODO_TOKEN=<your personal access token>     # never save it in a file
    python zenodo_upload.py                # creates a DRAFT (review it in the browser)
    python zenodo_upload.py --publish ID   # publishes draft ID -> permanent DOI

Uploads the paper PDF and the supplementary video, and fills in all metadata.
A published Zenodo record cannot be deleted, so check the draft first.
"""
import argparse, os, sys
import requests

API = "https://zenodo.org/api"
PDF = "paper/ral_friction_bound.pdf" if os.path.exists("paper/ral_friction_bound.pdf") else "ral_friction_bound.pdf"
VIDEO = "media/side_by_side_S1.mp4"

TITLE = ("Excitation-Aware Friction Capability Bounds for Uncertainty-Aware "
         "Speed Planning of Autonomous Ground Vehicles")
ABSTRACT = (
    "Speed planners must respect the tire–road friction limit, but friction is identifiable only when the tires "
    "work near saturation. We show that, for a saturating tire model, the Fisher information about friction in a "
    "measurement scales with the sixth power of tire utilisation, so ordinary driving leaves friction essentially "
    "unobserved and an estimate can remain unchanged long after the road has changed. Rather than trusting an "
    "estimate because it once converged, we propose an information-aware trust model: the friction used for "
    "planning is an approximate lower confidence bound whose width follows the information recently accumulated, "
    "relaxing toward a conservative prior when excitation is absent and with distance from the last informative "
    "excitation. A forward–backward friction-circle planner uses this value with a conditional feasibility "
    "guarantee. In simulation (20-run Monte Carlo, ±10% stiffness mismatch) a hidden friction drop caused lane "
    "departures in every run of certainty-equivalent planning and none with the proposed method, which recovered "
    "71% of the time lost by fixed conservative planning; a fixed safety margin on the estimate was either unsafe "
    "or slower, depending on how slippery the road became. In CARLA the method had no departures in 10 runs, "
    "whereas fixed-nominal planning departed in every run, certainty-equivalent planning in 17 of 18 and a fixed "
    "safety margin in 3 of 10, while being 2.6% faster than fixed conservative planning. We report the limits "
    "openly: the bound is approximate, is overconfident when the tire stiffness is underestimated, and in CARLA "
    "safety relied chiefly on the distance-based trust decay."
    "<br><br>Preprint; under review at IEEE Robotics and Automation Letters. "
    "Code, data and video: https://github.com/munavirzaman-git/friction-bound")

METADATA = {
    "metadata": {
        "upload_type": "publication",
        "publication_type": "preprint",
        "title": TITLE,
        "description": ABSTRACT,
        "creators": [{"name": "Zaman P K, Munavir", "affiliation": "Independent Researcher"}],
        "keywords": ["friction estimation", "speed planning", "autonomous vehicles",
                     "uncertainty", "persistent excitation", "CARLA"],
        "license": "cc-by-4.0",
        "access_right": "open",
        "language": "eng",
        "related_identifiers": [{
            "identifier": "https://github.com/munavirzaman-git/friction-bound",
            "relation": "isSupplementedBy",
            "resource_type": "software",
        }],
    }
}


def headers(token):
    return {"Authorization": f"Bearer {token}"}


def create_draft(token):
    for f in [PDF, VIDEO]:
        if not os.path.exists(f):
            sys.exit(f"missing file: {f}")
    r = requests.post(f"{API}/deposit/depositions", json={}, headers=headers(token))
    r.raise_for_status()
    dep = r.json()
    dep_id, bucket = dep["id"], dep["links"]["bucket"]
    print(f"created draft {dep_id}")
    for f in [PDF, VIDEO]:
        with open(f, "rb") as fp:
            u = requests.put(f"{bucket}/{os.path.basename(f)}", data=fp, headers=headers(token))
        u.raise_for_status()
        print(f"uploaded {f} ({os.path.getsize(f) / 1e6:.1f} MB)")
    m = requests.put(f"{API}/deposit/depositions/{dep_id}", json=METADATA, headers=headers(token))
    if m.status_code >= 400:
        print(m.text); m.raise_for_status()
    print("metadata set")
    print(f"\nREVIEW THE DRAFT HERE: https://zenodo.org/deposit/{dep_id}")
    print(f"If everything looks right, publish with:\n    python zenodo_upload.py --publish {dep_id}")


def publish(token, dep_id):
    r = requests.post(f"{API}/deposit/depositions/{dep_id}/actions/publish", headers=headers(token))
    if r.status_code >= 400:
        print(r.text); r.raise_for_status()
    d = r.json()
    print(f"PUBLISHED. DOI: {d.get('doi')}")
    print(f"Record: {d['links'].get('html', d['links'].get('record_html'))}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--publish", type=int, default=None, help="publish this draft id")
    args = ap.parse_args()
    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        sys.exit("set ZENODO_TOKEN first:  export ZENODO_TOKEN=...")
    if args.publish:
        publish(token, args.publish)
    else:
        create_draft(token)
