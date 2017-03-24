function restartService()
{
    var xmlHttp = new XMLHttpRequest();
    xmlHttp.open( "GET", "/restart", false ); // false for synchronous request
    xmlHttp.send( null );
    return xmlHttp.responseText;
}