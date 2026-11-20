"""CSS styling for HTML report exports."""

STYLING_CSS_FOR_HTML_EXPORT = """
<style>
    body {
        font-size: 18px !important;
    }
    .jp-Cell-outputWrapper, .jp-OutputArea-output, .output_area {
        display: flex;
        justify-content: safe center;
        max-width: 100%;
        overflow-x: auto;
    }
    .jp-RenderedHTMLCommon, .rendered_html, .text_cell_render {
        text-align: center;
        font-size: 1.1em !important;
    }
    table {
        margin-left: auto !important;
        margin-right: auto !important;
        font-size: 1em !important;
    }
    h1 { font-size: 2.5em !important; }
    h2 { font-size: 2em !important; }
    h3 { font-size: 1.6em !important; }
    p, td, th { font-size: 1.1em !important; }
    li { font-size: 1.1em !important; }
    ul, ol {
        text-align: left;
    }
    .output_png, .output_jpeg, .output_svg {
        display: flex;
        justify-content: safe center;
    }
    hr {
        border: none;
        border-top: 2px dotted #888;
        margin: 40px auto;
        width: 100%;
    }
    h2 {
        margin-top: 50px !important;
    }
</style>
"""
