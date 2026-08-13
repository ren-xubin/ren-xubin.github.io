# Writing a blog post

The site uses Jekyll, the Markdown blog engine supported by GitHub Pages.

## Create a post

1. Copy `_drafts/post-template.md`.
2. Rename the copy to `_posts/YYYY-MM-DD-short-title.md`.
3. Edit the title, date, summary, tags, and Markdown body.
4. Put post images under `assets/blog/short-title/`.
5. Commit and push. GitHub Pages will rebuild the blog automatically.

For example:

```text
_posts/2026-08-13-why-i-built-nanobot.md
```

The post will appear automatically on `/blog/`, newest first. Its public URL will be:

```text
/blog/2026/08/13/why-i-built-nanobot/
```

Keep unfinished writing in `_drafts/`; Jekyll does not publish that directory.

## Preview locally

Use Ruby 3.0 or newer. Install the dependencies once:

```bash
bundle install
```

Then start the local site:

```bash
bundle exec jekyll serve
```

Open `http://127.0.0.1:4000/blog/`.
