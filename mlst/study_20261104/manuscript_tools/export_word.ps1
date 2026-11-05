param([Parameter(Mandatory=$true)][string]$Stage,[string]$Suffix='',[string]$SourceSubdir='manuscript',[string[]]$Stems=@('Hydrogel_Continuation_MLST','Supplement_MLST','MLST_Cover_Letter'))
$ErrorActionPreference='Stop'
$qa=Join-Path 'H:\PolymerReview' ('wordqa_mlst_'+(Split-Path $Stage -Leaf)+$Suffix)
if (Test-Path -LiteralPath $qa) {throw 'QA directory already exists; preserve prior export'}
New-Item -ItemType Directory -Path $qa | Out-Null
$sources=$Stems
foreach($stem in $sources){Copy-Item -LiteralPath (Join-Path $Stage ($SourceSubdir+'\'+$stem+'.docx')) -Destination (Join-Path $qa ($stem+'.docx'))}
$preWord=@(Get-Process -Name WINWORD -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Id)
$word=$null;$owned=$false;$doc=$null;$records=@()
try {
 $word=New-Object -ComObject Word.Application
 $newWord=@(Get-Process -Name WINWORD -ErrorAction SilentlyContinue | Where-Object {$preWord -notcontains $_.Id})
 if($newWord.Count -ne 1){throw 'Cannot uniquely establish ownership of new Word process; no documents opened or settings changed'}
 $wordProcess=$newWord[0].Id
 if ($word.Documents.Count -ne 0) {throw 'New Word instance is not empty'}
 $owned=$true;$word.Visible=$false;$word.DisplayAlerts=0;$word.AutomationSecurity=3
 foreach($stem in $sources){
  $path=Join-Path $qa ($stem+'.docx');$pdf=Join-Path $qa ($stem+'.pdf')
  $before=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
  $doc=$word.Documents.Open($path,$false,$true)
  if (-not $doc.ReadOnly) {throw 'Word document is not read only'}
  $doc.ExportAsFixedFormat($pdf,17)
  $pages=$doc.ComputeStatistics(2)
  $doc.Close(0);[void][Runtime.InteropServices.Marshal]::ReleaseComObject($doc);$doc=$null
  $after=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
  if($before -ne $after) {throw 'DOCX bytes changed during read-only export'}
  $records+=[ordered]@{actual_utc=[DateTimeOffset]::UtcNow.ToString('o');stem=$stem;input_sha256=$before;pdf_sha256=(Get-FileHash -LiteralPath $pdf -Algorithm SHA256).Hash.ToLowerInvariant();pages=$pages;owned_word_process=$wordProcess;original_docx_resaved=$false}
 }
 $records | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $qa 'WORD_EXPORT_RECEIPT.json') -Encoding UTF8
} finally {
 if($null -ne $doc -and $owned){$doc.Close(0);[void][Runtime.InteropServices.Marshal]::ReleaseComObject($doc)}
 if($null -ne $word -and $owned){$word.Quit(0);[void][Runtime.InteropServices.Marshal]::ReleaseComObject($word)}
}
Get-Content -LiteralPath (Join-Path $qa 'WORD_EXPORT_RECEIPT.json') -Raw
