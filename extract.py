from docutils.core import publish_doctree
from docutils import nodes
from docutils.parsers.rst import roles

def dummy_role(name, rawtext, text, lineno, inliner, options=None, content=None):
    return [nodes.Text(text)], []

roles.register_local_role("doc", dummy_role)

rst = open("docs/source/user/first_mcdc.rst").read()

for node in publish_doctree(rst).findall(nodes.literal_block):
    if node.get("classes") == ['code', 'python3']:
        print(node.astext())

