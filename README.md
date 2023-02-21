# Neuro dashboards

A set of dashboards for visualizing neuroimaging data.

## Installation

1. Clone the repository

2. Install [`poetry`](https://python-poetry.org/) to `~/.local/bin`

    ```sh
    curl -sSL https://install.python-poetry.org | python3 -
    ```

3. Create a new virtual environment

    ```sh
    conda create -n neuro-dash python=3.9
    conda activate neuro-dash
    ```

4. Run `poetry install` from inside the repo

## Timeseries Viewer

```
python timeseries_viewer/app.py path/to/timeseries.npy
```

![Timeseries viewer](static/images/timeseries_viewer.gif)
