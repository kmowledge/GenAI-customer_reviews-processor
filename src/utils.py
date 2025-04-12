import csv

def save_results_to_csv(results, output_file):
    """Save processed review results to CSV file"""
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)

        # Write header
        if results and len(results) > 0:
            first_result = results[0]
            headers = [
                "review",
                "sentiment",
                "reasoning",
                "ground_truth",
                "satisfaction_score",
            ]

            if "message" in first_result:
                headers.append("message")

            if "recommended_trips" in first_result:
                headers.extend(
                    ["recommended_trip_1", "recommended_trip_2", "recommended_trip_3"]
                )

            if "discount_code" in first_result:
                headers.append("discount_code")

            writer.writerow(headers)

            # Write data rows
            for result in results:
                row = [
                    result["review"],
                    result["sentiment"],
                    result["reasoning"],
                    result["ground_truth"],
                    result["satisfaction_score"],
                ]

                if "message" in result:
                    row.append(result["message"])

                if "recommended_trips" in result:
                    trips = result["recommended_trips"]
                    for i in range(min(3, len(trips))):
                        row.append(
                            f"{trips[i]['destination']}: {trips[i]['description']}"
                        )

                    # Fill empty trip slots if less than 3 trips
                    for i in range(len(trips), 3):
                        row.append("")

                if "discount_code" in result:
                    row.append(result["discount_code"])

                writer.writerow(row)


def save_locations_to_csv(locations, output_file):
    """Save extracted locations to CSV file"""
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["review_id", "location", "entity_type"])

        for location in locations:
            writer.writerow(
                [location["review_id"], location["location"], location["entity_type"]]
            )