import shutil
import zipfile

SRC = "Corporate White Background.potx"
DST = "corporate_template.pptx"

TMPL_CT = (
    "application/vnd.openxmlformats-officedocument.presentationml.template.main+xml"
)
PRES_CT = (
    "application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"
)

with zipfile.ZipFile(SRC, "r") as zin:
    names = zin.namelist()
    with zipfile.ZipFile(DST, "w", zipfile.ZIP_DEFLATED) as zout:
        for name in names:
            data = zin.read(name)
            if name == "[Content_Types].xml":
                data = data.replace(TMPL_CT.encode(), PRES_CT.encode())
            zout.writestr(name, data)

print(f"Wrote {DST}")
