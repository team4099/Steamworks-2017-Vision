function reloadPage() {
	location.reload(true);
}

function restartService()
{
    setTimeout(reloadPage, 3000);
    jQuery.get("/restart", function(data) {
    	console.log("setting reload...");
    })
    
}