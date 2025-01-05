# Setting up LaTeX in VS Code (Windows)

## I. Introduction to LaTeX and Overleaf

- **LaTeX:** LaTeX is a powerful typesetting system widely used in academia, science, and engineering for creating high-quality documents, especially those containing complex mathematical formulas, tables, and figures. Unlike word processors like Microsoft Word, LaTeX focuses on the content and structure of the document, leaving the formatting to the system. You write plain text with specific commands (starting with `\`) that define the document's structure and elements, and LaTeX then compiles this into a beautifully formatted output (usually a PDF).
- **Overleaf:** Overleaf is an online LaTeX editor that provides a collaborative environment for writing and compiling LaTeX documents. It eliminates the need for local LaTeX installations and is excellent for collaborative projects. However, for offline work or more complex setups, a local installation is often preferred.

## II. Setting up LaTeX Locally in VS Code (This Guide)

### A. Installing the Necessary Software

1. **MiKTeX (LaTeX Distribution):** MiKTeX is a popular LaTeX distribution for Windows.

   - Download the MiKTeX basic installer from the official website: [https://miktex.org/](https://miktex.org/)
   - Run the installer _as administrator_. Choose to install for all users. Select "Install missing packages on-the-fly".
   - After installation, open the MiKTeX Console _as administrator_. Refresh the FNDB and Update Formats (in the "Tasks" tab). This step is often forgotten but very important!
2. **Strawberry Perl:** `latexmk`, a powerful build tool for LaTeX, is written in Perl.

   - Download the Strawberry Perl MSI installer from the official website: [http://strawberryperl.com/](http://strawberryperl.com/)
   - Run the installer.
   - Add the Perl `bin` directory (usually `C:\Strawberry\perl\bin` or `C:\Strawberry\c\bin`) to your system's PATH environment variable. This is a crucial step!
   - Restart your computer (or at least log out and back in) for the PATH changes to take effect.
3. **LaTeX Workshop Extension in VS Code:**

   - Open VS Code.
   - Go to the Extensions view (Ctrl+Shift+X).
   - Search for "LaTeX Workshop" and install it.

### B. Configuring LaTeX Workshop and Project

1. **Configure Output Directory:**

   - Open VS Code settings (File > Preferences > Settings).
   - Search for `latex-workshop.latex.outDir`.
   - Set the value to `"./build"`. This will output all generated LaTeX files to a `latex` subdirectory inside a `build` directory in your project's root.
   - Create a `build` directory and a `latex` subdirectory inside it in your project root folder.
2. **Install the `kvsetkeys` Package (If needed):** If you get an error about `kvsetkeys.sty` not being found, you need to install it:

   - Open the MiKTeX Console _as administrator_.
   - Go to the "Packages" tab, search for `kvsetkeys`, and install it.
   - Go to the "Tasks" tab and "Refresh file name database" and "Update Formats".
3. **.gitignore Configuration:** Create a `.gitignore` file in your project's root directory and add the following:

   ```
   build/
   ```

   This will prevent the build artifacts and `.npz` files from being tracked by Git.

### D. Building Your LaTeX Project

1. **Create your `main.tex` file:** This is your main LaTeX document.
2. **Create your `references.bib` file (if using a bibliography):** This file will contain your bibliographic entries.
3. **Compile:**
   - Use the LaTeX Workshop panel in VS Code (click the TeX icon and then "Build LaTeX project"

### F. Future Improvements (TODOs)

- **LaTeX Formatting:** Implement automatic LaTeX code formatting using `latexindent`. This will improve code readability and maintainability.
  - **Installation:** Install `latexindent` using `cpanm --notest App::latexindent` (requires Perl).
  - **VS Code Configuration:** Configure `latex-workshop.latexindent.path` in VS Code settings.
  - **Benefits:** Consistent formatting, improved readability.

## III. Converting LaTeX to Markdown (Optional)

You might sometimes want to convert your LaTeX documents to Markdown for better display on platforms like GitHub. Pandoc is a useful tool for this.

**Converting with Pandoc:**

1. **Install Pandoc:** Download and install Pandoc from the official website: [https://pandoc.org/](https://pandoc.org/)
2. **Navigate to Project Root:** Open a terminal window and navigate to your LaTeX project's root directory.
3. **Pandoc Command:** Run the following command:

   ```bash
   pandoc latex/main.tex -s -o README.md
   ```

   - `latex/main.tex`: Your main LaTeX document (relative to the project root).
   - `-s`: Creates a standalone HTML document (useful for previewing the Markdown output).
   - `-o README.md`: Output file name (in the project root).

**Addressing the Figures Path Issue**

During the conversion process, we encountered an issue with the paths to the figures. Because the `figures` directory is located inside the `latex` directory, Pandoc was not correctly locating the image files. As a temporary workaround, since there were only a few figures, we manually adjusted the image paths within the README.md file to be relative to the project root (add latex/).

Possible Soloution

use the `--resource-path` option correctly in the Pandoc command. This would avoid the need for manual path adjustments in the `main.tex` file. The correct usage is demonstrated in the Pandoc command above: `--resource-path=latex/figures`.

**Problems with Converting Underlining (`\underline{}`):**

Pandoc doesn't directly translate the LaTeX `\underline{}` command to standard Markdown. This means underlined text in your LaTeX source will not be underlined in the initial Markdown output.

Possible Soloution

Lua filter is a script that modifies Pandoc's internal representation of the document during conversion. This allows for custom transformations. To solve the `\underline{}` problem, a Lua filter can be used to convert the LaTeX command into equivalent HTML `<u>` tags, which _are_ supported in Markdown and render correctly on GitHub.

**Problems with Converting Algorithms (`algorithm` and `algorithmic` Environments):**

Pandoc's default LaTeX to Markdown conversion doesn't handle the `algorithm` and `algorithmic` environments well. These environments are used to typeset algorithms with structured formatting (using commands like `\REQUIRE`, `\ENSURE`, `\FOR`, `\WHILE`, `\STATE`, `\Comment`, etc.). Without a custom solution, Pandoc will often render the algorithm content as plain text, losing its intended structure and formatting.
