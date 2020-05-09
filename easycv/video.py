class Video:
    def __init__(self, path):
        self.path = path

    def _repr_html_(self):
        html = '<video width="720" height="480" controls>'
        html += '<source src="{}" type="video/mp4"></video>'.format(self.path)
        return html
