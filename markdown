
An explanation of markdown format

# Heading 1
## Heading 2
### Heading 3
#### Heading 4
##### Heading 5
###### Heading 6

# Paragraphs and Line Breaks

This is a normal paragraph:

    This is a code block.

This is another normal paragraph and end of code block.

# Horizontal Rules

Three or more...

---

Hyphens

***

Asterisks

___

Underscores

When you do that, you’ll see a horizontal rule line.

# Emphasis

*This text will be italic*
_This will also be italic_

**This text will be bold**
__This will also be bold__

_You **can** combine them_

# Lists

## Unordered

* Item 1
* Item 2
  * Item 2a
  * Item 2b

## Ordered

1. Item 1
1. Item 2
1. Item 3
   1. Item 3a
   1. Item 3b

# Images

![GitHub Logo](/images/logo.png)
Format: ![Alt Text](url)

This is an image ![Image of Yaktocat](https://octodex.github.com/images/yaktocat.png "Yaktocat")

# Links

http://github.com is automatically converted into a clickable link.
This is a link to [GitHub](http://github.com) as well.

# Blockquotes

As Kanye West said:

> We're living the future so
> the present is our past.

# Inline code

I think you should use an
`<addr>` element here instead.

# Syntax highlighting

```javascript
function fancyAlert(arg) {
  if(arg) {
    $.facebox({div:'#foo'})
  }
}
```

# Tables

First Header | Second Header
------------ | -------------
Content from cell 1 | Content from cell 2
Content in the first column | Content in the second column

# Task Lists

- [x] @mentions, #refs, [links](), **formatting**, and <del>tags</del> supported
- [x] list syntax required (any unordered or ordered list supported)
- [x] this is a complete item
- [ ] this is an incomplete item

# Emoji

@octocat :+1: This PR looks great - it's ready to merge! :shipit:

# GitHub Flavored Markdown

## Syntax highlighting

```javascript
function fancyAlert(arg) {
  if(arg) {
    $.facebox({div:'#foo'})
  }
}
```

## SHA references

Any reference to a commit’s SHA-1 hash will be automatically converted into a link to that commit on GitHub.

## Issue references within a repository

Any number that refers to an Issue or Pull Request will be automatically converted into a link.

## Username @mentions

Typing an @ symbol, followed by a username, will notify that person to come and view the comment. This is called an “@mention”, because you’re mentioning the individual. You can also @mention teams within an organization.

## Strikethrough

Any word wrapped with two tildes (like ~~this~~) will appear crossed out.




