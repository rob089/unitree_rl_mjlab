from mjlab.utils.lab_api.tasks.importer import import_packages

_BLACKLIST_PKGS = [
	"utils",
	".mdp",
	".a2",
	".as2",
	".go2",
	".h1_2",
	".h2",
	".r1",
]

import_packages(__name__, _BLACKLIST_PKGS)
