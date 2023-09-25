$hi=(dir *.html)
foreach ($i in $hi) {
	$otext = get-content $i
	$newtext = $otext -replace 'NAPE_Head.html', 'index.html'
	$newtext | set-content $i
}