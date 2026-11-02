# Project Instructions

Write clear, concise, and readable code.

Use Google-style formatting:

- Use clear function and variable names.
- Add comments or docstrings where they help explain non-obvious logic.
- Add simple comments only when they help explain the logic.

Keep solutions simple. This is not production code, so avoid extensive error
handling, logging, config files, or unnecessary abstractions.
Keep the code easy to understand, modify, and reuse.

When possible, display results clearly using pandas DataFrames or other simple
output methods. When you create figures or results, save them to `./result/` after creating
the directory if it does not exist, and call `plt.show()` only after saving.
The plots should have larger text and number should to be formatted with :, .

When available, modify the `.py` file and sync it to `.ipynb` file` using `Jupytext`.

When downloading new package, add this to the requirements file with the version. In default use conda do make the env and downloading. 

For each function make sure to have google-style description.

For each section have a title or subtitle in %%[markdown] cell.

The preferred pattern is to use four cells for each requirement: (1) a cell containing a function that computes the required results and returns them as a DataFrame, (2) a cell that calls the function, (3) a cell containing a function that plots the results, and (4) a cell that calls the plotting function. Repeat this structure consistently for every requirement. (e.g., compute_summary() → compute_summary(), plot_summary() → plot_summary()).