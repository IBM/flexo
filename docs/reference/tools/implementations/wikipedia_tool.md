The Wikipedia tool accepts lowercase language codes (including `simple`,
`be-tarask`, and `zh-min-nan`) and a non-empty page title. The configured endpoint
must use HTTPS on the selected language's exact `<lang>.wikipedia.org` host,
without credentials or a nonstandard port. Page titles are URL-encoded as a
single path segment. HTTP redirects are not followed; a redirected page request
returns an error and may need to be retried with its canonical title.

::: src.tools.implementations.wikipedia_tool.WikipediaTool
    options:
        show_root_heading: true
        show_source: true
        heading_level: 1
