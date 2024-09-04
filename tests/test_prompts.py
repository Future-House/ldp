import textwrap

from ldp.llms.prompts import indent_xml


def test_indent_xml():
    xml = "<root><child1>foo<child2>line1\nline2\nline3</child2></child1></root>"
    expected = textwrap.dedent(
        """\
        <root>
          <child1>
            foo
            <child2>
              line1
              line2
              line3
            </child2>
          </child1>
        </root>"""
    )
    assert indent_xml(xml) == expected
