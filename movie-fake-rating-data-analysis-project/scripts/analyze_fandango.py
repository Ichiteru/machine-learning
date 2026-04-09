from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    fandango = pd.read_csv(DATA_DIR / 'fandango_scrape.csv')
    all_sites = pd.read_csv(DATA_DIR / 'all_sites_scores.csv')
    return fandango, all_sites


def build_normalized_scores(fandango: pd.DataFrame, all_sites: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge(fandango, all_sites, on='FILM', how='inner')

    # Bring every source to a shared 0..5 scale so the averages are comparable.
    return pd.DataFrame(
        {
            'film': merged['FILM'],
            'fandango_stars': merged['STARS'],
            'fandango_rating': merged['RATING'],
            'rotten_tomatoes_critics': (merged['RottenTomatoes'] / 20).round(2),
            'rotten_tomatoes_users': (merged['RottenTomatoes_User'] / 20).round(2),
            'metacritic_critics': (merged['Metacritic'] / 20).round(2),
            'metacritic_users': (merged['Metacritic_User'] / 2).round(2),
            'imdb': (merged['IMDB'] / 2).round(2),
        }
    )


def fandango_summary(fandango: pd.DataFrame) -> dict[str, float]:
    voted = fandango[fandango['VOTES'] > 0].copy()
    voted['stars_diff'] = voted['STARS'] - voted['RATING']

    return {
        'movies_with_votes': int(len(voted)),
        'mean_stars': round(voted['STARS'].mean(), 3),
        'mean_rating': round(voted['RATING'].mean(), 3),
        'mean_diff': round(voted['stars_diff'].mean(), 3),
        'corr_rating_votes': round(fandango[['RATING', 'VOTES']].corr(numeric_only=True).loc['RATING', 'VOTES'], 3),
        'corr_stars_votes': round(fandango[['STARS', 'VOTES']].corr(numeric_only=True).loc['STARS', 'VOTES'], 3),
    }


def comparison_summary(norm_scores: pd.DataFrame) -> dict[str, object]:
    columns = [
        'fandango_stars',
        'fandango_rating',
        'rotten_tomatoes_critics',
        'rotten_tomatoes_users',
        'metacritic_critics',
        'metacritic_users',
        'imdb',
    ]
    worst_10 = norm_scores.nsmallest(10, 'rotten_tomatoes_critics')

    return {
        'matched_movies': int(len(norm_scores)),
        'mean_scores': norm_scores[columns].mean().round(2).to_dict(),
        'worst_10_mean_scores': worst_10[columns].mean().round(2).to_dict(),
    }


def print_section(title: str) -> None:
    print(f'\n{title}')
    print('-' * len(title))


def main() -> None:
    fandango, all_sites = load_data()
    norm_scores = build_normalized_scores(fandango, all_sites)

    local = fandango_summary(fandango)
    comparison = comparison_summary(norm_scores)

    print_section('Fandango Local Analysis')
    print(f"Movies with votes: {local['movies_with_votes']}")
    print(f"Mean STARS: {local['mean_stars']}")
    print(f"Mean RATING: {local['mean_rating']}")
    print(f"Mean STARS - RATING: {local['mean_diff']}")
    print(f"Correlation RATING vs VOTES: {local['corr_rating_votes']}")
    print(f"Correlation STARS vs VOTES: {local['corr_stars_votes']}")

    print_section('Cross-Site Comparison on a 0..5 Scale')
    print(f"Matched movies: {comparison['matched_movies']}")
    for key, value in comparison['mean_scores'].items():
        print(f'{key}: {value}')

    print_section('Worst 10 Movies by Rotten Tomatoes Critics')
    for key, value in comparison['worst_10_mean_scores'].items():
        print(f'{key}: {value}')

    print_section('Conclusion')
    print(
        'The dataset shows consistent signs that Fandango ratings are inflated relative '
        'to both its internal user rating and competing sites.'
    )


if __name__ == '__main__':
    main()
